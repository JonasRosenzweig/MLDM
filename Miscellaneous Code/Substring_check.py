### Substring match ###

OTHER_Headers = ['Washing','Ironing','Drying','Drycleaning','Bleaching','materiale','Materiale','Number','number','%',
                 'Type','type','Kolli','pct','Antal','Quantity','Qty','quantity','qty','Kode','kode','Fit','lukning',
                 'Category','category','Kategori','kategori','Weight','weight','Qty.','Customer','Units','Age','age',
                 'delivery','date','Date','Delivery','description','Description','Delivery','Season','Køn','Gender',
                 'Account','No.','no.',	'Stock','stock','pct.','month','Composition','composition','available','Finish',
                 'Order Closing','Washing','Beskrivelse 2', 'In Stock', 'Available','Date', 'Per Display', 'Weight',
                 'Division', 'Delivery', 'Brand', 'Drop']
PRICE_Headers = ['price','Price','Cost','VAT','Retail','Subtotal','Sub total','total','sub total','subtotal','RRP',
                 'Wholesale','wholesale','udsalgspris','pris','Pris','Udsalgspris','Katalogpris','katalogpris']
COLOR_Headers = ['Colour name', 'Color name', 'Color Name', 'Colour Name', 'color name']
SIZE_Headers = ['Size', 'size', 'Størrelse', 'størrelse', 'mål', 'Mål']
Headers_Rule_List = [OTHER_Headers, PRICE_Headers, COLOR_Headers, SIZE_Headers]

# check if substring in list
def Substring_in_list(string, l):
    if any(string in s for s in l):
        return True
    else:
        return False

# tests
#print('Testing SUBSTRING method', Substring_in_list('Retail Price', PRICE_Headers))
#print('Testing SUBSTRING method', Substring_in_list('Price', PRICE_Headers))
#print('Testing SUBSTRING method', Substring_in_list('Sub', PRICE_Headers))

def isSubString(string1, string2):
    if string1 in string2:
        return True
    else:
        return False

#print(isSubString('Sub', 'Subtotal'))


#matchers = ['abc','def']
#matching = [s for s in my_list if any(xs in s for xs in matchers)]

def Substring_match(match_list, string):
    for i in range(len(match_list)):
        if match_list[i] in string:
            return True


print(Substring_match(PRICE_Headers, 'Retail Price'))
