import os


def cleanup_folder(fp_folder_store):
    """
    Cleanup the contents of a folder or create it if it doesn't exist.

    Args:
        fp_folder_store (str): The absolute path of the folder to be cleaned up or created.

    Returns:
        None
    """
    if os.path.exists(fp_folder_store):
        # If folder exists, delete its contents
        for root, dirs, files in os.walk(fp_folder_store):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
    else:
        # If folder does not exist, create it
        os.makedirs(fp_folder_store)