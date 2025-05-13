def batchify(data, batch_size):
    """
    Split the data into batches of the given size.
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]