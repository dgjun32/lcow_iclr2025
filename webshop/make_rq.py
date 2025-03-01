import yaml

with open('environment.yml') as f:
    env = yaml.safe_load(f)

with open('requirements.txt', 'w') as f:
    for dep in env['dependencies']:
        if isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                f.write(f"{pip_dep}\n")