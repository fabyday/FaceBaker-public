def alphanum_comp(x, y):
    """
        check alphanum
        
        
        ex) 
            a10.obj < a11.obj
    """
    def token(x):
        re = []
        tmp = ""
        record = False
        record_digit = False
        for idx, c in enumerate(x) :
            if c.isdigit() :
                if record_digit == False and len(tmp)>0 : 
                    re.append(tmp)
                    tmp = ""   
                tmp += c
                record_digit = True 
            else : 
                if record_digit == True and len(tmp)>0 : 
                    re.append(int(tmp))
                    tmp = ""
                record_digit = False 
                tmp += c
        if len(tmp) > 0 :
            re.append(tmp)
        return re 

    x = token(x)
    y = token(y)
    m = min(len(x), len(y))
    for i in range(m):
        if x[i] == y[i]:
            continue
        elif x[i] < y[i]:
            return -1
        elif x[i] > y[i]:
            return 1
    return 0 