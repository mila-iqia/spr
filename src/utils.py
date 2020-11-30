import torch
import time
EPS = 1e-6
LOG_EPS = -13.8155


from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
from torch import nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class dummy_context_mgr:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def minimal_c51_loss(predictions, targets):
    """
    :param predictions: distribution for q(s, a) (from softmax).
    :param targets: targets for q(s, a)
    :return: cross entropy (loss)
    """
    predictions = torch.log(clamp_probs(predictions))
    loss = -torch.sum(targets*predictions, dim=-1)
    KL_div = torch.sum(targets *
        (torch.log(targets) - predictions), dim=-1)
    KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

    return loss, KL_div.detach()


def minimal_scalar_loss(predictions, targets, delta_clip=1):
    """
    :param predictions: Log q(s, a) (from logsoftmax or safe_log of softmax).
    :param targets: Log targets for q(s, a)
    :return: cross entropy (loss)
    """
    delta = predictions - targets
    value_losses = 0.5 * delta ** 2
    abs_delta = torch.abs(delta)

    if delta_clip is not None:  # Huber loss.
        b = delta_clip * (abs_delta - delta_clip / 2)
        value_losses = torch.where(abs_delta <= delta_clip, value_losses, b)
    td_abs_errors = abs_delta.detach()
    if delta_clip is not None:
        td_abs_errors = torch.clamp(td_abs_errors, 0, delta_clip)
    return value_losses, td_abs_errors


def average_targets(targets, weights):
    """
    :param targets: (batch, jumps, atoms).  Logits if distributional.
    :param lambdas: (batch,), in (0, 1).
    :return:
    """
    # Cut down the weights, in case we didn't search all the way.
    if targets.shape[1] != weights.shape[1]:
        weights = weights.clone()
        weights = weights[:, :targets.shape[1]]
        weights[:, -1] = weights[:, -1] + (1 - torch.sum(weights, -1, keepdim=False))

    targets = (targets*weights.unsqueeze(-1)).sum(-2)
    return targets


def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=tensor.device), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def safe_log(tensor):
    return torch.log(clamp_probs(tensor))


def clamp_probs(tensor):
    return torch.clamp(tensor, EPS, 1)


def c51_backup(n_step,
               returns,
               nonterminal,
               target_ps,
               select_action=False,
               V_max=10.,
               V_min=10.,
               n_atoms=51,
               discount=0.99,
               selection_values=None):

    z = torch.linspace(V_min, V_max, n_atoms, device=target_ps.device)

    if select_action:
        if selection_values is None:
            selection_values = target_ps
        target_qs = torch.tensordot(selection_values, z, dims=1)  # [B,A]
        next_a = torch.argmax(target_qs, dim=-1)  # [B]
        target_ps = select_at_indexes(next_a.to(target_ps.device), target_ps)  # [B,P']

    delta_z = (V_max - V_min) / (n_atoms - 1)
    # Make 2-D tensor of contracted z_domain for each data point,
    # with zeros where next value should not be added.
    next_z = z * (discount ** n_step)  # [P']
    next_z = torch.ger(nonterminal, next_z)  # [B,P']
    ret = returns.unsqueeze(1)  # [B,1]
    next_z = torch.clamp(ret + next_z, V_min, V_max)  # [B,P']

    z_bc = z.view(1, -1, 1)  # [1,P,1]
    next_z_bc = next_z.unsqueeze(1)  # [B,1,P']
    abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
    projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.

    # projection_coeffs is a 3-D tensor: [B,P,P']
    # dim-0: independent data entries
    # dim-1: base_z atoms (remains after projection)
    # dim-2: next_z atoms (summed in projection)

    target_ps = target_ps.unsqueeze(1)  # [B,1,P']
    target_p = (target_ps * projection_coeffs).sum(-1)  # [B,P]
    target_p = torch.clamp(target_p, EPS, 1)
    return target_p


def scalar_backup(n, returns, nonterminal, qs, discount, select_action=False, selection_values=None):
    """
    :param qs: q estimates
    :param n: n-step
    :param nonterminal:
    :param returns: Returns, already scaled by discount/nonterminal
    :param discount: discount in [0, 1]
    :return:
    """
    if select_action:
        if selection_values is None:
            selection_values = qs
        next_a = selection_values.mean(-1).argmax(-1)
        qs = select_at_indexes(next_a, qs)
    while len(returns.shape) < len(qs.shape):
        returns = returns.unsqueeze(-1)
    while len(nonterminal.shape) < len(qs.shape):
        nonterminal = nonterminal.unsqueeze(-1)
    discount = discount ** n
    try:
        qs = nonterminal*qs*discount + returns
    except:
        import ipdb;
        ipdb.set_trace()
    return qs


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights


def get_augmentation(augmentation, imagesize):
    if isinstance(augmentation, str):
        augmentation = augmentation.split("_")
    transforms = []
    for aug in augmentation:
        if aug == "affine":
            transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
        elif aug == "rrc":
            transformation = RandomResizedCrop((imagesize, imagesize), (0.8, 1))
        elif aug == "blur":
            transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
        elif aug == "shift" or aug == "crop":
            transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
        elif aug == "intensity":
            transformation = Intensity(scale=0.05)
        elif aug == "none":
            continue
        else:
            raise NotImplementedError()
        transforms.append(transformation)

    return transforms


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


def maybe_transform(image, transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * image
        return processed_images


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def minimal_quantile_loss(pred_values, target_values, taus, kappa=1.0):
    if len(pred_values.shape) == 3:
        output_shape = pred_values.shape[:2]
        target_values = target_values.expand_as(pred_values)
        pred_values = pred_values.flatten(0, 1)
        target_values = target_values.flatten(0, 1)
    else:
        output_shape = pred_values.shape[:1]

    if pred_values.shape[0] != taus.shape[0]:
        # somebody has added states along the batch dimension,
        # probably to do multiple timesteps' losses simultaneously.
        # Since the standard in this codebase is to put time on dimension 1 and
        # then flatten 0 and 1, we can do the same here to get the right shape.
        expansion_factor = pred_values.shape[0]//taus.shape[0]
        taus = taus.unsqueeze(1).expand(-1, expansion_factor, -1,).flatten(0, 1)

    td_errors = pred_values.unsqueeze(-1) - target_values.unsqueeze(1)
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    loss = batch_quantile_huber_loss.squeeze(1)

    # Just use the regular loss as the error for PER, at least for now.
    return loss.view(*output_shape), loss.detach().view(*output_shape)


def renormalize(tensor, first_dim=-3):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)

