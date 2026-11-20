"""Strict state validation shared by real adapters."""


def strict_state_load(module, state, component):
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = [{"key": key, "expected": list(expected[key].shape), "checkpoint": list(state[key].shape)}
              for key in sorted(set(expected) & set(state)) if expected[key].shape != state[key].shape]
    if missing or unexpected or shapes:
        raise RuntimeError(f"Strict checkpoint mismatch in {component}: missing={missing}; unexpected={unexpected}; shape_mismatches={shapes}")
    module.load_state_dict(state, strict=True)
    return {"component": component, "keys": len(state), "strict": True}
