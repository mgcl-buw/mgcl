import sys

# Accumulate timings for init vs solve test, i.e. calculating solve from oh+init+solve 
# etc.

if sys.argv == 0:
    print("Usage: accumulate_benchs [file1.csv] [file2.csv] ...")
    exit(1)

for f in sys.argv:
    if f == sys.argv[0]:
        continue

    print("Processing " + str(f) + " ...")

    t_oh = -1
    t_init = -1
    t_init_ocl = -1

    total_append = "\n\n"

    with open(f, "r+") as file:
        for line in file:
            l = line.rstrip().split(";")

            # skip header
            if "title" in l[0]:
                continue

            name = l[1]
            time = float(l[4])

            if "init+oh+solve" in name:
                t_solve = time - t_oh - t_init

                # append lines
                app = name.split(",")
                app = ",".join(app[0:3])

                # accumulate values and append to file
                app_init = app + ", init\""
                app_l = l
                app_l[1] = app_init
                app_l[4] = str(t_init)
                total_append += ";".join(app_l) + "\n"
                # file.write("\n" + ";".join(app_l) + "\n")

                app_init = app + ", solve\""
                app_l = l
                app_l[1] = app_init
                app_l[4] = str(t_solve)
                total_append += ";".join(app_l) + "\n"
                # file.write(";".join(app_l) + "\n")

                if t_init_ocl != -1:
                    app_init = app + ", init_ocl\""
                    app_l = l
                    app_l[1] = app_init
                    app_l[4] = str(t_init_ocl)
                    total_append += ";".join(app_l) + "\n"
                    # file.write(";".join(app_l) + "\n")

                # reset timings
                t_init = -1
                t_solve = -1
                t_init_ocl = -1
                t_oh = -1

            elif "init+oh" in name:
                t_init = time - t_oh
            elif "init ocl env" in name:
                t_init_ocl = time - t_oh
            elif "overhead (oh)" in name:
                t_oh = time

        file.write(total_append)

