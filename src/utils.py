def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class dummy_context_mgr:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

