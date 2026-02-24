from PIL import Image


def is_image_openable(image_path): # helper function to drop broken images
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def add_drive_path(path):
    if path.startswith('/media/my_drives/DATA4/data/image_benchmark_phi') == False:
        return f'/media/my_drives/DATA4/data/image_benchmark_phi{path}'
    else:
        return path