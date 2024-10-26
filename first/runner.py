import subprocess


if __name__ == '__main__':
    def format_stdout(self):
        return self[2:-5]  # remove b' and '\r\n

    with open("results.txt", "w") as out:
        for file in ["data_2.txt", "data_4.txt", "data_5.txt"]:
            out.write(f"QR dec -- {file}:\n")
            for i in range(0, 11):
                out.write(format_stdout(str(subprocess.run(f"python ./main.py {file} {i} -qr", capture_output=True).stdout)) + "\n")
            out.write("\n")

        out.write("\n")

        for file in ["data_2.txt", "data_4.txt", "data_5.txt"]:
            out.write(f"NE method -- {file}:\n")
            for i in range(0, 11):
                out.write(format_stdout(str(subprocess.run(f"python ./main.py {file} {i} -ne", capture_output=True).stdout)) + "\n")
            out.write("\n")