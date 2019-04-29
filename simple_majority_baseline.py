

if __name__ == "__main__":
    trainfilename = "data/train.data"
    testfilename = "data/test.data"
    outfilename = "outputs/simple-majority-baseline-output.txt"

    ntline = 0
    true_count = 0
    false_count = 0
    with open(trainfilename) as tf:
        for tline in tf:
            tline = tline.strip()
            line = tline.split('\t')
            if len(line) == 7:
                tuples = line[4]
                numbers = tuples.replace("(", "").replace(")", "").split(", ")
                num_0 = int(numbers[0])
                num_1 = int(numbers[1])
                if num_0 > num_1:
                    true_count += 1
                else:
                    false_count += 1

    with open(testfilename) as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 7:
                ntline +=1
    # output the results into a file
    outf = open(outfilename, 'w')

    for x in range(ntline):

        if true_count > false_count:

            outf.write("true\t" + "{0:.4f}".format(0) + "\n")
        # outf.write("true\t" + "0.0000" + "\n")
        else:
            outf.write("false\t" + "{0:.4f}".format(0) + "\n")
        # outf.write("false\t" + "0.0000" + "\n")

    outf.close()