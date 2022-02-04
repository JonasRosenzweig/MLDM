from stdnum import ean

def validate_EAN(string):
    try:
        ean.validate(string)
        print('Is EAN')
    except:
        print('Not EAN')

validate_EAN('5711529602921')
validate_EAN('5711529600000')
validate_EAN('191241147248')