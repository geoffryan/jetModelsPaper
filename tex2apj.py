import errno
import os
import sys
import subprocess


def apjify(inName, outDir):

    if not os.path.isfile(inName):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                inName)

    inDir, name = os.path.split(inName)

    if inDir == '':
        inDir = '.'

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

    with open(inName, 'r') as f:
        inTxt = f.read()
    f.close()

    filelist = getFigFilenames(inTxt)
    trans = makeTranslator(filelist)

    process(inTxt, trans, inDir, outDir, name)


def process(inTxt, trans, inDir, outDir, texName):

    outName = os.path.join(outDir, texName)

    # Make text with replaced figure names
    outTxt = inTxt
    for name in trans:
        outTxt = outTxt.replace(name, trans[name])

    # Write the new file
    with open(outName, 'w') as f:
        f.write(outTxt)
    f.close()

    # Copy the figure files
    for name in trans:
        figpath = os.path.join(inDir, name)
        newfigpath = os.path.join(outDir, trans[name])
        print("Copy {0:s} to {1:s}".format(figpath, newfigpath))
        subprocess.run(["cp", figpath, newfigpath])


def getFigFilenames(src):

    inFig = False

    beginFig = r'\begin{figure'
    endFig = r'\end{figure'
    figDir = 'figs/'

    filelist = []
    filenames = []

    for i, line in enumerate(src.split('\n')):

        # Trim comments
        lineout = line.split('%')[0]
        if lineout == '':
            continue

        # Look for figure filenames
        if inFig:
            if endFig in lineout:
                filelist.append(filenames)
                inFig = False
            else:
                matches = findFilenamesInLine(line, figDir)
                filenames.extend(matches)
        else:
            if beginFig in lineout:
                filenames = []
                inFig = True

    return filelist


def findFilenamesInLine(line, figDir):

    start = 0
    matches = []
    while figDir in line[start:]:
        a = line.find(figDir, start)
        b = line.find('}', a)
        match = ''.join(line[a:b].split())
        matches.append(match)
        start = b

    return matches


def makeTranslator(fileList):

    trans = {}

    subfiglabel = 'abcdefghijklmnopqrstuvwxyz'

    for i, names in enumerate(fileList):
        if len(names) == 1:
            _, ext = os.path.splitext(names[0])
            trans[names[0]] = "fig{0:d}{1:s}".format(i+1, ext)
        elif len(names) > 1:
            for j, name in enumerate(names):
                _, ext = os.path.splitext(name)
                trans[name] = "fig{0:d}{1:s}{2:s}".format(i+1, subfiglabel[j],
                                                          ext)

    return trans


if __name__ == "__main__":

    Nargs = 2

    usage = "usage: $ python3 inputfilename outputdirectory"

    if len(sys.argv) < Nargs+1:
        print("Incorrect arguments: " + str(sys.argv[1:]))
        print(usage)
        sys.exit(1)

    apjify(sys.argv[1], sys.argv[2])
