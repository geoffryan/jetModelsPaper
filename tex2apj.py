import errno
import os
import sys


def apjify(inName, outDir):

    if not os.path.isfile(inName):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                inName)

    inDir, name = os.path.split(inName)

    if inDir == '':
        inDir = os.getcwd()

    if os.path.exists(outDir):
        if not os.path.isdir(outDir):
            e = errno.ENOTDIR
            raise NotADirectoryError(e, os.strerror(e), outDir)
    else:
        os.mkdir(outDir)

    if os.path.samefile(inDir, outDir):
        msg = "Output directory {0} contains input file {1}"\
            .format(outDir, inName)
        raise RuntimeError(msg)

    outName = os.path.join(outDir, name)

    with open(inName, 'r') as f:
        inTxt = f.readlines()
    f.close()

    with open(outName, 'w') as f:
        parseTex(inTxt, f, inDir, outDir)
    f.close()


def parseTex(src, f, inDir, outDir):

    inFig = False
    inSubFig = False
    fignum = 0
    subfignum = 0

    beginFig = '\\begin{figure'
    endFig = '\\end{figure'

    for line in src:

        # Deal with comments
        lineout = line.split('%')[0]
        if lineout == '':
            continue
        if lineout[-1] != '\n':
            lineout += '\n'

        f.write(lineout)


if __name__ == "__main__":

    Nargs = 2

    usage = "usage: $ python3 inputfilename outputdirectory"

    if len(sys.argv) < Nargs+1:
        print("Incorrect arguments: " + str(sys.argv[1:]))
        print(usage)
        sys.exit(1)

    apjify(sys.argv[1], sys.argv[2])
