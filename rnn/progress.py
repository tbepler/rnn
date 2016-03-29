import time
import sys

def progress_bar(out=sys.stdout, delta=2.0, bar_len=40):
    clear_code = ''
    tstart = time.time()
    tprev = tstart
    while True:
        header,progress,extras = yield
        tcur = time.time()
        if progress == 1 or progress == 0: #this signals done
            out.write(clear_code)
            print >>out, header,
            write_extras(extras, out)
            print >>out, ''
            clear_code = ''
            tstart = time.time()
            tprev = tstart
            continue
        if tcur-tprev >= delta:
            tprev = tcur
            out.write(clear_code)
            print >>out, header,
            write_extras(extras, out)
            print >>out, '' #newline
            n = int(progress*bar_len)
            bar = ''.join(['#']*n + [' ']*(bar_len-n))
            eta = (tcur-tstart)/progress*(1-progress)
            hours, rem = divmod(eta, 3600)
            mins, secs = divmod(rem, 60)
            print >>out, '    [{}] {:7.2%}, eta {:0>2}:{:0>2}:{:0>2}'.format(bar, progress
                                                                             , int(hours)
                                                                             , int(mins)
                                                                             , int(secs))
            out.flush()
            clear_code = '\033[1F\033[K\033[1F\033[K'

def write_extras(extras, out):
    first = True
    for k,v in extras.iteritems():
        if not first:
            out.write(',')
        out.write(' {}={}'.format(k,v))
        first = False
            
if __name__ == '__main__':
    bar = progress_bar()
    total = 1e7
    next(bar)
    for i in xrange(int(total)):
        frac = i/total
        bar.send(('Testing:', frac, {'total':total, 'cur':i}))
    bar.send(('Done', 1, {'total':total}))
    total = 1e7
    for i in xrange(int(total)):
        frac = i/total
        bar.send(('Testing:', frac, {'total':total, 'cur':i}))
    bar.send(('Done', 1, {'total':total}))
