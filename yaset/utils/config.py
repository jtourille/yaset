

def replace_auto(options: dict = None):
    for key, value in options.items():
        if isinstance(value, dict):
            replace_auto(options=value)
        else:
            if value == "auto":
                options[key] = -1
