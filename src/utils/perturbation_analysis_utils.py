
def is_input_file_processed(input_file, preexisting_output_files):
    is_present = False

    for f in preexisting_output_files:
        if input_file in f:
            is_present = True
            break
    return is_present
