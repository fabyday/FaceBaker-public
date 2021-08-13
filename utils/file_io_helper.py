import itertools as it, glob


# this module help file io.
# wrapping glob module.





def iglob_multiple_file_types(*patterns, **kwargs):
    """ 
        see https://stackoverflow.com/questions/4568580/python-glob-multiple-filetypes
        it used lazy evaluation.
        
        pattern : glob path argument lists. ex) [ "*.ext", ... , "*.ppl" ]
        kwargs : 
            recursive : default = False

        Use Case :
            for filename in multiple_file_types("*.txt", "*.sql", "*.log"):
                # do stuff
    """
    recursive = kwargs.get("recursive", False)

    return it.chain.from_iterable(glob.iglob(pattern, recursive=recursive) for pattern in patterns)




def glob_multiple_file_types(*patterns, **kwargs):
    """
        pattern : glob path argument lists. ex) [ "*.ext", ... , "*.ppl" ]
        kwargs : 
            recursive : default = False
        ===================================================================
        return [glob(pattern1) + glob(pattern2) + .... glob(pattern_n)]
    """
    recursive = kwargs.get("recursive", False)
    
    return [path for patt in patterns for path in glob.glob(patt, recursive=recursive)]
