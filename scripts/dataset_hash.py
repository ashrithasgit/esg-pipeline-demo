# scripts/dataset_hash.py
import hashlib, sys
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/dataset_hash.py <path>')
        sys.exit(1)
    print(sha256_file(sys.argv[1]))
