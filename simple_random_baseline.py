import random

if __name__ == "__main__":
    trainfilename = "data/train.data"
    testfilename = "data/test.data"
    outfilename = "outputs/simple-baseline.output"

    ntline = 0
    with open(testfilename) as tf:
        for tline in tf:
            tline = tline.strip()
            if len(tline.split('\t')) == 7:
                ntline += 1

    # output the results into a file
    with open(outfilename, 'w') as outf:
        for x in range(ntline):
            score = random.random()
            if score >= 0.5:
                outf.write("true\t" + "{0:.4f}".format(score) + "\n")
            else:
                outf.write("false\t" + "{0:.4f}".format(score) + "\n")
        outf.close()
