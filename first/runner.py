import subprocess
from sys import argv

if __name__ == '__main__':
    def format_stdout(self):
        return self[2:-5]  # remove b' and '\r\n

    file_name = "results." + "csv" if argv[-1] == "-gencsv" else "txt"

    with open(file_name, "w") as out:
        header = "N; cond type; cond value; t (sec); NRMSE;\n" if argv[-1] == "-gencsv" else ""
        out.write(header)
        for file in ["data_2.txt", "data_4.txt", "data_5.txt"]:
            header2 = f"QR decomposition; {file};" + ";" * 3 + "\n" if argv[-1] == "-gencsv" else f"QR dec -- {file}:\n"
            out.write(header2)

            for i in range(0, 10):
                out.write(format_stdout(str(subprocess.run(f"python ./main.py {file} {i} -qr {'-gencsv' if argv[-1] == '-gencsv' else ''}", capture_output=True).stdout)) + "\n")
            out.write("\n")

        out.write("\n")

        for file in ["data_2.txt", "data_4.txt", "data_5.txt"]:
            header2 = f"Normal equations; {file};" + ";" * 3 + "\n" if argv[-1] == "-gencsv" else f"QR dec -- {file}:\n"
            out.write(header2)

            for i in range(0, 10):
                out.write(format_stdout(str(subprocess.run(f"python ./main.py {file} {i} -ne {'-gencsv' if argv[-1] == '-gencsv' else ''}", capture_output=True).stdout)) + "\n")
            out.write("\n")