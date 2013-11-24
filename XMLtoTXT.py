import re


def stripxml(data):
    pattern = re.compile(r'<.*?>')
    return pattern.sub('', data)


def main():
    data = open('Resume1.txt').read()
    #print data

    text_data = stripxml(data)
    print text_data


if __name__ == '__main__':
    main()