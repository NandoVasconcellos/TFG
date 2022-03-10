import argparse



class Test():
    #def handler(self, parser):
    #    args = parser.parse_args()
    #    print(args)

    def __init__(self, args: dict):
        print(self)
        for i in args:
            print(i)


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='This app will extract features from your matlab files',
        epilog='Enjoy the program! :)'
    )

    # Required positional argument
    parser.add_argument('--folder', type= str,
                        help='Complete path to folder where there are all files from a user')
                        
    # Required positional argument
    parser.add_argument('--user', type=str,
                        help='Name of user existing in the file name. E.g. 0091, user#0091...')

    #convert args to dictionary
    dict_args = dict()
    for arg in parser.parse_args()._get_kwargs():
        dict_args[arg[0]] = arg[1]

    Test( dict_args )