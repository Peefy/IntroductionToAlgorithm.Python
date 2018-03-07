
if __name__ == '__main__':
    import chapter6_1 as _chapter6_1
    import chapter6_2 as _chapter6_2
    import chapter6_3 as _chapter6_3
    import chapter6_4 as _chapter6_4
else:
    from . import chapter6_1 as _chapter6_1
    from . import chapter6_2 as _chapter6_2
    from . import chapter6_3 as _chapter6_3
    from . import chapter6_4 as _chapter6_4

chapter6_1 = _chapter6_1.Chapter6_1()
chapter6_2 = _chapter6_2.Chapter6_2()
chapter6_3 = _chapter6_3.Chapter6_3()
chapter6_4 = _chapter6_4.Chapter6_4()

def chapter6_printall():
    '''
    print CRLS 6.1 - 6.4 note
    '''
    chapter6_1.note()
    chapter6_2.note()
    chapter6_3.note()
    chapter6_4.note()

if __name__ == '__main__':
    chapter6_printall()
else:
    pass
