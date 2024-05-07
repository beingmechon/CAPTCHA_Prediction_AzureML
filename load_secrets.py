import configparser

def load_secrets(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    secrets = {}
    for section in config.sections():
        for key, value in config.items(section):
            secrets[key] = value
    return secrets


if __name__ == '__main__':
    secrets = load_secrets('secrets.ini')
    print(secrets)
