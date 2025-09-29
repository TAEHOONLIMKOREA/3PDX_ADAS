import configparser
import os

def load_config():
    parser = configparser.ConfigParser()
    # 현재 스크립트가 위치한 디렉터리 경로를 가져옴
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, "settings.ini") 
    # print(config_file)
    # 파일이 존재하고 파일인지 (디렉터리가 아님) 확인
    if os.path.exists(config_file) and os.path.isfile(config_file):
        with open(config_file, encoding='utf-8') as fp:
            parser.read_file(fp)
        redis_config = {
            'host': parser.get('redis', 'host'),
            'port': parser.getint('redis', 'port'),
            'db': parser.getint('redis', 'db'),
            'uri': parser.get('redis', 'uri')
        }

        postgres_config = {
            'uri': parser.get('postgres', 'uri'),
            'host': parser.get('postgres', 'host'),
            'port': parser.getint('postgres', 'port'),
            'user': parser.get('postgres', 'user'),
            'password': parser.get('postgres', 'password'),
            'db': parser.get('postgres', 'db')
        }

        mongo_config = {
            'host': parser.get('mongo', 'host'),
            'port': parser.getint('mongo', 'port'),
            'user': parser.get('mongo', 'user'),
            'password': parser.get('mongo', 'password'),
            'authSource': parser.get('mongo', 'authSource'),
            'authMechanism': parser.get('mongo', 'authMechanism')
        }

        minio_config = {
            'endpoint': parser.get('minio', 'endpoint'),
            'access_key': parser.get('minio', 'access_key'),
            'secret_key': parser.get('minio', 'secret_key'),
            'secure': parser.getboolean('minio', 'secure')
        }

        return {
            'redis': redis_config,
            'postgres': postgres_config,
            'mongo': mongo_config,
            'minio': minio_config
        }
    else:
        # 파일이 없을 경우 명확한 에러메시지 또는 기본값을 반환하도록 할 수 있음
        raise FileNotFoundError(f"Config file not found: {config_file}")

CONFIG = load_config()
print(CONFIG)