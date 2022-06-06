import time

def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1


def main(num_interruptions=3):
    num_interruptions_counter = 0
    for i in infinite_sequence():
        try:
            print(i)
        except KeyboardInterrupt as e:
            num_interruptions_counter += 1
            if num_interruptions == num_interruptions_counter:
                break
            else:
                print(e, f'Сделай так еще {num_interruptions-num_interruptions_counter} раз')
                time.sleep(3)
                continue

if __name__ == '__main__':
        main()
