
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-dk', '--data_kind', type=str, choices=['test', 'train', 'valid'],
                    help='specify whether the file being processed is training, validation or testing dataset')
parser.add_argument('-n', '--num_context', type=int, metavar='',
                    help='number of context turns to include in processed dataset')
parser.add_argument('-df', '--data_file', type=str, required=True,
                    help="The input data file (a text file) to process.")
args = parser.parse_args()


def process(raw_file, data_type, num_context=None):

    # 'test.txt' is 'train.history_belief.txt' in case of training data or 'val.history_belief.txt' in case of validation data
    with open(raw_file, 'r') as f:
        with open(data_type + '.input_context.txt', 'w') as fr1:
            with open(data_type + '.target_belief.txt', 'w') as fr:

                def create_files(line):
                    delimiter = '<|belief|>'
                    split_line = line.split(delimiter)
                    split_line[1] = '<|endoftext|>' + delimiter + split_line[1]
                    split_line[0] = split_line[0] + '<|endoftext|>'
                    fr.write(split_line[1])
                    fr.write('\n')
                    fr1.write(split_line[0])
                    fr1.write('\n')
                    fr1.write('\n')

                if num_context is not None:
                    for line in f:
                        if line.count("<|user|>") <= num_context:
                            create_files(line)
                        else:
                            split_line1 = line.split('<|user|>')
                            context_list = split_line1[-num_context:]
                            for i in range(len(context_list)):
                                context_list[i] = '<|user|>' + context_list[i]
                            context_line1 = "".join(context_list)
                            context_line = '<|endoftext|> <|context|> ' + context_line1

                            create_files(context_line)
                else:
                    for line in f:
                        create_files(line)


def main():
    file_to_process = args.data_file
    # specify file type to train, test or valid
    file_type = args.data_kind

    if args.num_context is None:
        process(file_to_process, data_type=file_type)

    if args.num_context is not None:
        process(file_to_process, num_context=args.num_context,
                data_type=file_type)


if __name__ == "__main__":
    main()
