import os

from argparser import parse_arguments
from utils import *


def main():
    args = parse_arguments()
    print("*****************************")
    print(args)
    print("*****************************")

    fix_seed(args.random_seed)

    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass

    total = 0
    correct_list = []
    for i, data in enumerate(dataloader):
        print("*************************")
        print("{}st data".format(i + 1))

        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()

        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x
        else:
            raise ValueError("method is not properly defined ...")

        # Answer prediction by generating text ...
        max_length = (
            args.max_length_cot if "cot" in args.method else args.max_length_direct
        )
        z = decoder.decode(args, x, max_length, i, 1)

        # Answer extraction for zero-shot-cot ...
        if args.method == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)
            print(z2 + pred)
        else:
            pred = z
            print(x + pred)

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)

        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print("*************************")

        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1  # np.array([y]).size(0)

        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))


if __name__ == "__main__":
    main()
