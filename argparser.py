from typing import Optional

from tap import Tap


class ArgumentParser(Tap):
    api_log_file_name: Optional[
        str
    ] = None  # mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]
    random_seed: Optional[int] = 1

    """
    [
        "aqua",
        "gsm8k",
        "commonsensqa",
        "addsub",
        "multiarith",
        "strategyqa",
        "svamp",
        "singleeq",
        "bigbench_date",
        "object_tracking",
        "coin_flip",
        "last_letters",
    ]
    """
    dataset: Optional[str] = "aqua"  # dataset used for experiment

    minibatch_size: Optional[
        int
    ] = 1  # minibatch size should be 1 because GPT-3 API takes only 1 input for each request

    max_num_worker: Optional[int] = 3  # maximum number of workers for dataloader

    """
    ["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"]
    """
    model: Optional[
        str
    ] = "gpt3"  # model used for decoding. Note that 'gpt3' are the smallest models.

    """
    ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"]
    """
    method: Optional[str] = "zero_shot_cot"

    cot_trigger_no: Optional[
        int
    ] = 1  # "A trigger sentence that elicits a model to execute chain of thought"

    max_length_cot: Optional[
        int
    ] = 128  # maximum length of output tokens by model for reasoning extraction

    max_length_direct: Optional[
        int
    ] = 32  # maximum length of output tokens by model for answer extraction

    """
    whether to limit test dataset size. 
    if 0, the dataset size is unlimited and 
    we use all the samples in the dataset for testing
    """
    limit_dataset_size: Optional[int] = 10

    api_time_interval: Optional[float] = 1.0
    log_dir: Optional[str] = "./log/"  # log directory

    # Additional parameters
    dataset_path: Optional[str]
    direct_answer_trigger: Optional[str]
    direct_answer_trigger_for_zeroshot: Optional[str]
    direct_answer_trigger_for_zeroshot_cot: Optional[str]
    direct_answer_trigger_for_fewshot: Optional[str]
    cot_trigger: Optional[str]
    plausible_answer_trigger: Optional[str]


def parse_arguments():
    # parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    # parser.add_argument(
    #     "--api_log_file_name",
    #     type=str,
    #     default=None,
    #     help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]",
    # )

    # parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="aqua",
    #     choices=[
    #         "aqua",
    #         "gsm8k",
    #         "commonsensqa",
    #         "addsub",
    #         "multiarith",
    #         "strategyqa",
    #         "svamp",
    #         "singleeq",
    #         "bigbench_date",
    #         "object_tracking",
    #         "coin_flip",
    #         "last_letters",
    #     ],
    #     help="dataset used for experiment",
    # )

    # parser.add_argument(
    #     "--minibatch_size",
    #     type=int,
    #     default=1,
    #     choices=[1],
    #     help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request",
    # )

    # parser.add_argument(
    #     "--max_num_worker",
    #     type=int,
    #     default=3,
    #     help="maximum number of workers for dataloader",
    # )

    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     default="gpt3",
    #     choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"],
    #     help="model used for decoding. Note that 'gpt3' are the smallest models.",
    # )

    # parser.add_argument(
    #     "--method",
    #     type=str,
    #     default="zero_shot_cot",
    #     choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"],
    #     help="method",
    # )
    # parser.add_argument(
    #     "--cot_trigger_no",
    #     type=int,
    #     default=1,
    #     help="A trigger sentence that elicits a model to execute chain of thought",
    # )
    # parser.add_argument(
    #     "--max_length_cot",
    #     type=int,
    #     default=128,
    #     help="maximum length of output tokens by model for reasoning extraction",
    # )
    # parser.add_argument(
    #     "--max_length_direct",
    #     type=int,
    #     default=32,
    #     help="maximum length of output tokens by model for answer extraction",
    # )
    # parser.add_argument(
    #     "--limit_dataset_size",
    #     type=int,
    #     default=10,
    #     help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing.",
    # )
    # parser.add_argument("--api_time_interval", type=float, default=1.0, help="")
    # parser.add_argument("--log_dir", type=str, default="./log/", help="log directory")

    # args = parser.parse_args()

    args = ArgumentParser().parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = (
            "Choose the most plausible answer from among choices A through E."
        )
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")

    return args
