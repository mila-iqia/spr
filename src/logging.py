import numpy as np
import wandb


def reset_trackers(jumps):
    trackers = dict()
    trackers["epoch_losses"] = np.zeros(jumps + 1)
    trackers["value_losses"] = np.zeros(jumps + 1)
    trackers["policy_losses"] = np.zeros(jumps + 1)
    trackers["reward_losses"] = np.zeros(jumps + 1)
    trackers["nce_losses"] = np.zeros(jumps + 1)
    trackers["nce_accs"] = np.zeros(jumps + 1)
    trackers["value_errors"] = np.zeros(jumps + 1)
    trackers["reward_errors"] = np.zeros(jumps + 1)
    trackers["mean_pred_values"] = np.zeros(jumps + 1)
    trackers["mean_pred_rewards"] = np.zeros(jumps + 1)
    trackers["mean_target_values"] = np.zeros(jumps + 1)
    trackers["mean_target_rewards"] = np.zeros(jumps + 1)
    trackers["pred_entropy"] = np.zeros(jumps + 1)
    trackers["target_entropy"] = np.zeros(jumps + 1)
    trackers["iterations"] = np.zeros(1)

    return trackers


def update_trackers(trackers,
                    reward_losses,
                    nce_losses,
                    nce_accs,
                    policy_losses,
                    value_losses,
                    epoch_losses,
                    value_errors,
                    reward_errors,
                    pred_values,
                    pred_rewards,
                    target_values,
                    target_rewards,
                    pred_entropy,
                    target_entropy):
    trackers["iterations"] += 1
    trackers["nce_losses"] += np.array(nce_losses)
    trackers["nce_accs"] += np.array(nce_accs)
    trackers["reward_losses"] += np.array(reward_losses)
    trackers["policy_losses"] += np.array(policy_losses)
    trackers["value_losses"] += np.array(value_losses)
    trackers["epoch_losses"] += np.array(epoch_losses)
    trackers["value_errors"] += np.array(value_errors)
    trackers["reward_errors"] += np.array(reward_errors)
    trackers["mean_pred_values"] += np.array(pred_values)
    trackers["mean_pred_rewards"] += np.array(pred_rewards)
    trackers["mean_target_values"] += np.array(target_values)
    trackers["mean_target_rewards"] += np.array(target_rewards)
    trackers["pred_entropy"] += np.array(pred_entropy)
    trackers["target_entropy"] += np.array(target_entropy)


def summarize_trackers(trackers):
    iterations = trackers["iterations"]
    nce_losses = np.array(trackers["nce_losses"] / iterations)
    nce_accs = np.array(trackers["nce_accs"] / iterations)
    reward_losses = np.array(trackers["reward_losses"] / iterations)
    policy_losses = np.array(trackers["policy_losses"] / iterations)
    value_losses = np.array(trackers["value_losses"] / iterations)
    epoch_losses = np.array(trackers["epoch_losses"] / iterations)
    value_errors = np.array(trackers["value_errors"] / iterations)
    reward_errors = np.array(trackers["reward_errors"] / iterations)
    pred_values = np.array(trackers["mean_pred_values"] / iterations)
    pred_rewards = np.array(trackers["mean_pred_rewards"] / iterations)
    target_values = np.array(trackers["mean_target_values"] / iterations)
    target_rewards = np.array(trackers["mean_target_rewards"] / iterations)
    pred_entropy = np.array(trackers["pred_entropy"] / iterations)
    target_entropy = np.array(trackers["target_entropy"] / iterations)

    return nce_losses, nce_accs, reward_losses, value_losses, policy_losses, \
           epoch_losses, value_errors, reward_errors, pred_values, \
           target_values, pred_rewards, target_rewards, pred_entropy, \
           target_entropy


def log_results(trackers,
                steps,
                prefix='train',
                verbose_print=True,):

    iterations = trackers["iterations"]
    if iterations == 0:
        # We did nothing since the last log, so just quit.
        return

    nce_losses, nce_accs, reward_losses, value_losses, policy_losses, \
    epoch_losses, value_errors, reward_errors, pred_values, target_values,\
    pred_rewards, target_rewards, pred_entropies, target_entropies = summarize_trackers(trackers)

    print(
        "{} Epoch: {}, Epoch L.: {:.3f}, NCE L.: {:.3f}, NCE A.: {:.3f}, Rew. L.: {:.3f}, Policy L.: {:.3f}, Val L.: {:.3f}, Rew. E.: {:.3f}, P. Rews {:.3f}, T._Rs. {:.3f}, Val E.: {:.3f}, P. Vs. {:.3f}, T._Vs. {:.3f},  P. Ents. {:.3f}, T. Ents. {:.3f}".format(
            prefix.capitalize(),
            steps,
            np.mean(epoch_losses),
            np.mean(nce_losses),
            np.mean(nce_accs),
            np.mean(reward_losses),
            np.mean(policy_losses),
            np.mean(value_losses),
            np.mean(reward_errors),
            np.mean(pred_rewards),
            np.mean(target_rewards),
            np.mean(value_errors),
            np.mean(pred_values),
            np.mean(target_values),
            np.mean(pred_entropies),
            np.mean(target_entropies),
        ))

    for i in range(len(epoch_losses)):
        jump = i
        if verbose_print:
            print(
                "{} Jump: {}, Epoch L.: {:.3f}, NCE L.: {:.3f}, NCE A.: {:.3f}, Rew. L.: {:.3f}, Policy L.: {:.3f}, Val L.: {:.3f}, Rew. E.: {:.3f}, P. Rs. {:.3f}, T._Rs. {:.3f}, Val E.: {:.3f}, P. Vs. {:.3f}, T._Vs. {:.3f},  P. Ents. {:.3f}, T. Ents. {:.3f}".format(
                    prefix.capitalize(),
                    jump,
                    epoch_losses[i],
                    nce_losses[i],
                    nce_accs[i],
                    reward_losses[i],
                    policy_losses[i],
                    value_losses[i],
                    reward_errors[i],
                    pred_rewards[i],
                    target_rewards[i],
                    value_errors[i],
                    pred_values[i],
                    target_values[i],
                    pred_entropies[i],
                    target_entropies[i]))

        wandb.log({prefix + 'Jump {} loss'.format(jump): epoch_losses[i],
                   prefix + 'Jump {} NCE loss'.format(jump): nce_losses[i],
                   prefix + 'Jump {} NCE acc'.format(jump): nce_accs[i],
                   prefix + "Jump {} Reward Loss".format(jump): reward_losses[i],
                   prefix + 'Jump {} Value Loss'.format(jump): value_losses[i],
                   prefix + "Jump {} Reward Error".format(jump): reward_errors[i],
                   prefix + "Jump {} Policy loss".format(jump): policy_losses[i],
                   prefix + "Jump {} Value Error".format(jump): value_errors[i],
                   prefix + "Jump {} Pred Rewards".format(jump): pred_rewards[i],
                   prefix + "Jump {} Target Rewards".format(jump): target_rewards[i],
                   prefix + "Jump {} Pred Values".format(jump): pred_values[i],
                   prefix + "Jump {} Target Values".format(jump): target_values[i],
                   prefix + "Jump {} Pred Entropies".format(jump): pred_entropies[i],
                   prefix + "Jump {} Target Entropies".format(jump): target_entropies[i],
                   'FM epoch': steps})

    wandb.log({prefix + ' loss': np.mean(epoch_losses),
               prefix + ' NCE loss': np.mean(nce_losses),
               prefix + ' NCE acc': np.mean(nce_accs),
               prefix + " Reward Loss": np.mean(reward_losses),
               prefix + ' Value Loss': np.mean(value_losses),
               prefix + " Reward Error": np.mean(reward_errors),
               prefix + " Policy loss": np.mean(policy_losses),
               prefix + " Value Error": np.mean(value_errors),
               prefix + " Pred Rewards".format(jump): np.mean(pred_rewards),
               prefix + " Pred Values".format(jump): np.mean(pred_values),
               prefix + " Target Rewards".format(jump): np.mean(target_rewards),
               prefix + " Target Values".format(jump): np.mean(target_values),
               prefix + " Pred Entropies".format(jump): np.mean(pred_entropies),
               prefix + " Target Entropies".format(jump): np.mean(target_entropies),
               'FM epoch': steps})

