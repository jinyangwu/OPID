#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from utils import chat_completion_with_retry, create_openai_client, extract_message_text


REPO_ROOT = Path(__file__).resolve().parent


def load_env_file(env_file: str) -> Dict[str, str]:
    """Load a shell-style .env file into os.environ.

    This supports the common lines used by `source .env`, including:
      export KEY=value
      KEY="value"
      KEY='value'
    Values from the file overwrite the current process environment, matching
    the behavior of running `source .env` in a shell.
    """
    loaded: Dict[str, str] = {}
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = REPO_ROOT / env_path
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value
        loaded[key] = value
    return loaded


def mask_secret(value: Optional[str], keep: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep * 2) + value[-keep:]


prompt = '''Analyze the following agent episode and return ONLY valid JSON.\n\nYou need to complete all three fields:\n1. Write a concise episode_summary.\n2. Write one episode_hint that extracts the failed trajectory into avoidance rules: the core mistake and warning signs that agent should avoid. \n3. Provide concise, action-oriented decision guidance for at most 5 critical step(s) from the candidate set as entries in step_hints; use the full episode to infer the guidance, but phrase each hint as advice the policy can act on at that step.\n\nImportant constraints:\n- Step indexing is 0-based: step 0 is the first step of the trajectory.\n- Use the task description together with the episode context to judge progress and mistakes.\n- Use the full episode context to identify what each critical step should have done better.\n- Each step_hints value should be one short imperative sentence for the policy at that step.\n- Write step_hints as policy-facing guidance, not as retrospective explanation of the trajectory.\n- Return only these top-level fields: episode_summary, episode_hint, step_hints.\n- The chosen steps are exactly the keys present in step_hints.\n\nReturn format:\n{\n  \"episode_summary\": \"string\",\n  \"episode_hint\": \"string\",\n  \"step_hints\": {\n    \"0\": \"hint for step 0\",\n    \"2\": \"hint for step 2\"\n  }\n}\n\nEpisode context:\n- Task description: clean some soapbar and put it in garbagecan.\n- episode_success: failure\n- Candidate step indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]\n- Interaction trajectory: Step 0\nObservation: -= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a shelf 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, a towelholder 2, and a towelholder 1.\n\nYour task is to: clean some soapbar and put it in garbagecan.\nResponse: <think>\nGiven the task to clean some soapbar and put it in the garbagecan, it's likely that a soapbar is somewhere in the room that needs to be retrieved. I will start by checking the closets, cabinets, and sinkbasins, as these are common places for soapbars. Since there isn't a direct action for washing soapbar, I will assume that there is a soapbar available in one of the cabinets or on a countertop or shelf. \n\nI will begin by asking for the inventory to check if there's already a soapbar.\n</think>\n<action>inventory</action>\nReturn: 0.000000\nStep 1\nObservation: You are not carrying anything.\nResponse: <think>\nBased on the current observation, I am in a room that contains various objects and one of the desired tasks is to clean some soapbar and put it in the garbagecan. Given that I have not taken any action yet, the best course of action is to find where a soapbar might be located. Based on the presence of a sinkbasin (sinkbasin 1 and sinkbasin 2), it is possible that a soapbar could be there. I should therefore go there to check and then, if I find a soapbar, clean it before placing it in the garbagecan.\n</think>\n<action>go to sinkbasin 1</action>\nReturn: 0.000000\nStep 2\nObservation: You arrive at sinkbasin 1. On the sinkbasin 1, you see nothing.\nResponse: <think>\nSince the first action did not yield any soapbar, I should continue exploring the available locations where soapbar might be found. Looking at my previous observation and the admissible actions, the topmost places where soapbar might be found are the cabinets and the sinkbasins. Given that the sinkbasins did not have the soapbar, I should check the cabinets next. 'Cabinet 1' and 'Cabinet 2' are directly accessible. Additionally, the countertop 1 is another possible location.\n\nI'll start by checking the cabinets, then move on to the countertop.\n</think>\n<action>go to cabinet 1</action>\nReturn: 0.000000\nStep 3\nObservation: You arrive at cabinet 1. The cabinet 1 is closed.\nResponse: <think>\nBased on the observations, I have arrived at cabinet 1, which is closed. The task is to find a soapbar and put it in the garbagecan. However, as the cabinet is closed, there is no immediate indicator of a soapbar inside. I should first try to open the cabinet to check if there is soapbar there, and then find the garbagecan if the soapbar is found to complete the task.\n</think>\n<action>open cabinet 1</action>\nReturn: 0.000000\nStep 4\nObservation: You open the cabinet 1. The cabinet 1 is open. In it, you see a spraybottle 2.\nResponse: <think>\nBased on the current situation, I have opened cabinet 1 and found a spraybottle 2 inside. Cleaning a soapbar requires a clear workspace, which I already have due to the lack of clutter at sinkbasin 1. The soapbar must be found in either the cabinet, bathtubbasin, or countertop. I will check the bathtubbasin given it might be easier to find the soapbar if it has already been used or placed there.\n\nAfter opening cabinet 1, I should examine the remaining cabinets and countertops to see if the soapbar is present.\n</think>\n<action>examine cabinet 2</action>\nReturn: 0.000000\nStep 5\nObservation: Nothing happens.\nResponse: <think>\nBased on the current observation, it seems that the cabinet 1 does not contain the soapbar. The previous actions of opening cabinet 1 and examining it 2 times did not reveal the soapbar. I need to check some other locations such as the cabinet 2, cabinet 3, and sinkbasins where soapbars might be located.\n</think>\n<action>go to cabinet 2</action>\nReturn: 0.000000\nStep 6\nObservation: You arrive at cabinet 2. The cabinet 2 is closed.\nResponse: <think>\nGiven the current situation, I first need to recognize that soapbar items are not likely to be found within the referred cabinets, bath fixtures, countertops, towel holders, or sink basins. Since the cabinet 1 was assessed to contain a spraybottle, it is prudent to check cabinet 2 to potentially find a soapbar as well.\n\nMy next action should be to open cabinet 2 and assess if there is a soapbar inside. \n</think>\n<action>open cabinet 2</action>\nReturn: 0.000000\nStep 7\nObservation: You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.\nResponse: <think>\nBased on the previous actions and observations, I have already checked cabinet 1 and cabinet 2, and found there were no soapbar or other necessary items needed to complete this task. Therefore, I will now check the remaining cabinet.\n\nThere are cabinets 1 and 2 (already checked), cabinet 3, cabinet 4, bathtubbasin 1, countertop 1, garbagecan 1, handtowelholder 1, shelf 1, sinkbasin 1, sinkbasin 2, toilet 1, toiletpaperhanger 1, towelholder 1, and towelholder 2.\n\nCabinet 3 and Cabinet 4 are potential places where the soapbar could be. Check cabinet 3 next.\n</think>\n<action>go to cabinet 3</action>\nReturn: 0.000000\nStep 8\nObservation: You arrive at cabinet 3. The cabinet 3 is closed.\nResponse: <think>\nGiven the current situation, I need to first address the fact that no clean soap bar is available. Since the cabinets contain a spraybottle, the soap bar should likely be in cabinet 1 or cabinet 3. However, cabinet 1 has already been checked and contained only a spraybottle. \n\nCabinet 3 is closed, so I should first check this cabinet to see if the soap bar is inside. After that, I need to go to sinkbasin 1 to clean the soap bar once I find it.\n\nIf I open cabinet 3, I may find the soap bar there.\n</think>\n<action>open cabinet 3</action>\nReturn: 0.000000\nStep 9\nObservation: You open the cabinet 3. The cabinet 3 is open. In it, you see a spraybottle 3.\nResponse: <think>\nGiven the current observation, cabinet 3 is open and we see a spraybottle 3. None of the previous actions resembled cleaning soapbar or putting it in the garbagecan. The most action-oriented task based on the current observation is to check if there's any soapbar, which would likely be stored elsewhere. Head to the bathroom areas for such items. Start by examining cabinet 3 again if necessary.\n</think>\n<action> examine cabinet 3 </action>\nReturn: 0.000000\nStep 10\nObservation: The cabinet 3 is open. In it, you see a spraybottle 3.\nResponse: <think>\nGiven my current task to clean a soapbar and put it in the garbagecan, I need to locate the soapbar first. The recent observations mentioned a spraybottle, but no soapbar was seen. The cabinet 2 and cabinet 3 did not contain any soapbar. The toilet 1 and the cabinets 1, 4, and 2 were also not found to contain a soapbar. Therefore, cabinet 1 seems to be the most plausible place to find the soapbar. After checking cabinet 1, I should then be able to either take the soapbar and proceed to clean it, or look for a way to clean it (if it’s not already clean).\n</think>\n<action> go to cabinet 1 </action>\nReturn: 0.000000\nStep 11\nObservation: You arrive at cabinet 1. The cabinet 1 is open. In it, you see a spraybottle 2.\nResponse: <think>\nGiven the current situation, I have already checked cabinet 1, cabinet 2, and cabinet 3 but did not find the soapbar. The spraybottle 2 is present in cabinet 1, but it's not the target item. I should check the other cabinets and areas where soapbars are typically stored before moving on to the countertops or garbage areas. I'll start by checking cabinet 4.\n</think>\n<action>go to cabinet 4</action>\nReturn: 0.000000\nStep 12\nObservation: You arrive at cabinet 4. The cabinet 4 is closed.\nResponse: <think>\nBased on the current observations, I have visited cabinets 1, 2, 3, and just encountered cabinet 4. The previous actions showed that a spraybottle was found in cabinet 3 and cabinet 1, but no soapbar. Cabinet 2 didn't reveal anything when previously explored. \n\nCabinet 4, currently closed, has not yet been explored, and given the task to clean soapbar and put it in the garbage can, I need to investigate cabinet 4 first to find the soapbar. Once I find it, I can move to the garbage can to dispose of it.\n</think>\n<action>open cabinet 4</action>\nReturn: 0.000000\nStep 13\nObservation: You open the cabinet 4. The cabinet 4 is open. In it, you see a cloth 3, and a soapbottle 1.\nResponse: <think> 我目前在 cabinet 4 里看到一个布料和一罐肥皂，目标是清理这块肥皂然后把它放进垃圾桶。当前步骤是清理肥皂。我已经打开了柜子 4，现在需要选择接下来的步骤。根据任务，下一步应该是先取走柜子里的肥皂。所以我应该采取的动作是“take soapbottle 1 from cabinet 4”。然后可以进行清理肥皂的步骤，最后将清理后的肥皂放入垃圾桶。 </think>\n<action> take soapbottle 1 from cabinet 4 </action>\nReturn: 0.000000\nStep 14\nObservation: You pick up the soapbottle 1 from the cabinet 4.\nResponse: <think>\nBased on the previous steps, we have already opened and inspected cabinets 3 and 4, taken a soapbottle 1 from cabinet 4, and are now holding the soapbottle 1. The task is to clean the soapbar and put it in the garbagecan. Given that the soapbar is typically stored in the cabinet with the soapbottle, it's likely cabinet 4 has the soapbar as well. We have already taken the soapbottle, so the next logical step is to find and clean the soapbar, and then dispose of the used soapbottle in the garbagecan.\n\nI should now examine the cabinet 4 to find the soapbar and take it to the sinkbasin where I can clean it. If there’s no soapbar in cabinet 4, I will need to search somewhere else - likely the cabinet at the top of the list might be a better possibility given it's the original purpose of the soapbar 1 being taken.\n\nLet's check cabinet 4 first.\n</think>\n<action>\nexamine cabinet 4\n</action>\nReturn: 0.000000\nStep 15\nObservation: The cabinet 4 is open. In it, you see a cloth 3.\nResponse: <think>\nBased on the recent observations, the cabinet 4 still contains a cloth 3 and a soapbottle 1. Since the goal is to clean the soapbar and put it in the garbagecan, the soapbottle 1 should be put back in the cabinet 4 where it was found. After cleaning the soapbar, subsequent steps will include moving it to the garbagecan.\n\nFirst, I should put the soapbottle back in the cabinet 4. After that, it's time to attend to the actual task of cleaning the soapbar and then dispose of it.\n</think>\n<action>move soapbottle 1 to cabinet 4</action>\nReturn: 0.000000\nStep 16\nObservation: You move the soapbottle 1 to the cabinet 4.\nResponse: <think>\nBased on the current situation, I have taken the soapbottle 1 from cabinet 4 and moved it to the cabinet 4. Now, I need to find something to clean the soapbar. The cabinet 1 had a spraybottle 2, which could be useful for wiping the soapbar. I should take the spraybottle 2, go to the bathtubbasin 1, and wipe the soapbar. After that, I will take the soapbar to the garbagecan 1 to dispose of it properly.\n</think>\n<action>take spraybottle 2 from cabinet 1</action>\nReturn: 0.000000\nStep 17\nObservation: Nothing happens.\nResponse: <think>\nBased on the observations and actions taken so far, I need to first decide where to take the soapbar from. The relevant steps available are 'go to cabinet 1', 'go to bathtubbasin 1', and 'go to sinkbasin 1', since a soapbar is typically found in those locations. After taking the soapbar, I will then head to the garbagecan to dispose of the used soapbar.\n\nGiven the chance to examine cabinet 4 again, I will do so to ensure that there's no soapbar still there. I will also check cabinet 1 (although I suspect there's no soapbar there), the sinkbasin 1, and the bathtubbasin 1. \n\nOnce I identify where the soapbar is, I will take it to the sinkbasin 1 to clean the soapbar first (I imagine I might need a cloth to wipe it or use the water there), then move the cleaned soapbar to the sinkbasin 1 to be put in the garbage next.\n</think>\n<action> go to cabinet 1 </action>\nReturn: 0.000000\nStep 18\nObservation: You arrive at cabinet 1. The cabinet 1 is open. In it, you see a spraybottle 2.\nResponse: <think>\nGiven the current situation, I have already taken the spraybottle 2 from cabinet 1. My next logical step would be to locate the soapbar. Having already seen soap in cabinet 4, I will check there first. If it's not there, I then need to find it elsewhere, if needed, by taking an inventory of other locations. \n\nI will choose to go to cabinet 4 to check for the soapbar.\n</think>\n<action> go to cabinet 4 </action>\nReturn: 0.000000\nStep 19\nObservation: You arrive at cabinet 4. The cabinet 4 is open. In it, you see a cloth 3, and a soapbottle 1.\nResponse: <think>\nBased on the previous observations and actions, I now have a soapbottle 1 in my hand. The next logical step would be to examine the soapbottle to confirm its contents and ensure it's safe to use. Following that, I'll take the soap from the soapbottle and move it to the corrected environment, which seems to be located in some sort of bathroom facilities. My final action will be to place the soap in the garbagecan. \n</think>\n<action> take cloth 3 from cabinet 4 </action>\nReturn: 0.000000\nStep 20\nObservation: You pick up the cloth 3 from the cabinet 4.\nResponse: <think>\nGiven the current situation, it seems I need to clean the soapbar first before placing it in the garbagecan. Since I have taken the cloth from the cabinet 4, the next logical step is to go to the bathtubbasin 1 which is typically where you clean soapbar and other items. Once cleaned, I need to place the cloth back in cabinet 4, as it seems that is the closest place for a clean item. After cleaning, I will take the soapbottle and proceed to the garbagecan to dispose of the soap.\n</think>\n<action> go to bathtubbasin 1 </action>\nReturn: 0.000000\nStep 21\nObservation: You arrive at bathtubbasin 1. On the bathtubbasin 1, you see a cloth 2, and a dishsponge 1.\nResponse: <think>\nGiven my current observation, I see that on the bathtubbasin 1, there is a cloth 2 and a dishsponge 1, but no soap or soapbar. I need to find where the soapbar is and also find a garbagecan for disposing of the used cloth. The discovery of the soapbar is my top priority to achieve the task of cleaning the soapbar and putting it in the garbagecan. Since the soapbar was last seen in the cabinet 4, I should check there first. If I cannot find the soapbar in cabinet 4, then it might be on the countertop 1 or in the garbagecan 1, which I can check next.\n</think>\n<action>go to cabinet 4</action>\nReturn: 0.000000\nStep 22\nObservation: You arrive at cabinet 4. The cabinet 4 is open. In it, you see a soapbottle 1.\nResponse: <think>\nBased on the previous observations, it seems that the soapbar is not accessible in the current cabinet 4. The last available location where the soapbar is reported to be is cabinet 1. Therefore, the first step should be to go to cabinet 1 to search for the soapbar. If it's not there, I will move on to the other locations mentioned.\n</think>\n<action>go to cabinet 1</action>\nReturn: 0.000000\nStep 23\nObservation: You arrive at cabinet 1. The cabinet 1 is open. In it, you see a spraybottle 2.\nResponse: <think>\nBased on the observations, it seems that the soapbar is still in the cabinet 4. The current step is about cleaning the soapbar. I should search cabinet 4 for the soapbar, then apply the soapbar to the cloth, and finally place the soaped cloth in the garbagecan.\n\nFirst, I need to find the soapbar in cabinet 4.\n</think>\n<action> go to cabinet 4 </action>\nReturn: 0.000000\nStep 24\nObservation: You arrive at cabinet 4. The cabinet 4 is open. In it, you see a soapbottle 1.\nResponse: <think>\nGiven the observations, I have already picked up and discarded a cloth 3, examined some items in the bathtubbasin 1, and visited cabinets 1 and 4. My next significant action should be to clean the soapbottle 1 I have. Then, based on the observations, the soapbottle 1 is in cabinet 4, where it can be collected.\n\nAfterwards, the next logical step would be to take the soapbar to the garbagecan to dispose of it properly as per the task. This step should be followed by cleaning the cabinet 4 to be prepared for the next action.\n</think>\n<action> go to cabinet 4 </action>\nReturn: 0.000000\nStep 25\nObservation: Nothing happens.\nResponse: <think>\nBased on the previous observations, it seems that the cabinet 4 contains soapbottles. I need to find a soapbar first. The previous actions were directed towards finding a soapbar, so I will check the cabinet 4 again, as soapbars are not commonly found in other locations I've visited.\n\nIf I can't find the soapbar in cabinet 4, I might need to check the bathtubbasin 1, countertop 1, or possibly another cabinet.\n\nI will check cabinet 4 once more.\n</think>\n<action> go to cabinet 4 </action>\nReturn: 0.000000\nStep 26\nObservation: Nothing happens.\nResponse: <think>\nGiven the observations, it appears that I have already checked cabinet 4 multiple times and have not found the soapbar there. Since I am not sure which cabinet the soapbar might be in, I should go through all the cabinets one by one to find it. I can also check the countertop and shelf, as these are likely common places to find soap. I will start by going to cabinet 1.\n\nAfter searching all possible locations, if I still cannot find the soapbar, I will check each cabinet again but in that order (cabinet 4, cabinet 3, cabinet 2, cabinet 1). If that still failed, I will go to the sink for a soap, as soap isn’t necessarily in cabinets.\n</think>\n<action>go to cabinet 1</action>\nReturn: 0.000000\nStep 27\nObservation: You arrive at cabinet 1. The cabinet 1 is open. In it, you see a spraybottle 2.\nResponse: <think>\nObservation 27 shows that the cabinet 1 still contains a spraybottle 2. Since I need to clean a soapbar and put it in the garbagecan, I should first check if there is a soapbar in the cabinet. Then, I will need to clean the soapbar and dispose of it in the garbagecan. Given that I am at cabinet 1, I should first go to the cabinet 4 to check for the soapbar. If I still cannot find a soapbar, I will have to look for it elsewhere, perhaps in a different cabinet or the countertop.\n</think>\n<action>go to cabinet 4</action>\nReturn: 0.000000\nStep 28\nObservation: You arrive at cabinet 4. The cabinet 4 is open. In it, you see a soapbottle 1.\nResponse: <think>\nGiven the current situation, I need to get soapbar to clean and dispose of it. I've already gone to cabinet 1 and found a spraybottle 2, but there's nothing new in cabinet 4 or cabinet 1. I should check the countertop 1 and shelf 1 for the soapbar, and after that, I'll go to the garbagecan 1 to dispose of it. \n</think>\n<action> go to countertop 1 </action>\nReturn: 0.000000\nStep 29\nObservation: You arrive at countertop 1. On the countertop 1, you see a mirror 1, and a showerglass 1.\nResponse: <think>\nGiven the recent observations, it seems like the soapbottle in cabinet 4 is empty. After examining the countertop 1, there is no soapbar or soapbottle visible. I should check the bathroom cabinets and garbage can next. My current objective is to find and clean some soapbar, so I will start by checking cabinet 1, as it was the previous location of the soapbottle we found empty.\n</think>\n<action>go to cabinet 1</action>\nReturn: 0.000000
'''
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test an OpenAI-compatible API after loading .env.")
    parser.add_argument("--env-file", default=".env", help="Env file to load first. Default: .env")
    parser.add_argument("--api-key", default=None, help="Override OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL.")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL.")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: okasdasdas",
        help="Prompt used for the connectivity test.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Max completion tokens for the test request. Default: 64",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Client timeout in seconds. Default: 60",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Retry attempts. Defaults to OPENAI_API_RETRIES or 3 after loading .env.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=None,
        help="Initial retry delay. Defaults to OPENAI_API_RETRY_DELAY or 1.0 after loading .env.",
    )
    parser.add_argument(
        "--dump-response",
        action="store_true",
        help="Print raw response JSON after a successful request.",
    )
    return parser.parse_args()


def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
        "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL"),
        "model": args.model or os.environ.get("OPENAI_MODEL"),
        "prompt": args.prompt,
        "max_completion_tokens": args.max_completion_tokens
        if args.max_completion_tokens is not None
        else int(os.environ.get("OPENAI_API_MAX_COMPLETION_TOKENS", "64")),
        "temperature": args.temperature,
        "timeout": args.timeout
        if args.timeout is not None
        else float(os.environ.get("OPENAI_API_TIMEOUT", "60")),
        "retries": args.retries
        if args.retries is not None
        else int(os.environ.get("OPENAI_API_RETRIES", "3")),
        "retry_delay": args.retry_delay
        if args.retry_delay is not None
        else float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0")),
        "dump_response": args.dump_response,
    }


def print_error(exc: BaseException) -> None:
    print("\nRequest failed.", file=sys.stderr)
    print(f"Exception type: {type(exc).__name__}", file=sys.stderr)
    print(f"Exception: {exc}", file=sys.stderr)

    cause = exc.__cause__ or exc
    if cause is not exc:
        print(f"caused_by: {type(cause).__name__}: {cause}", file=sys.stderr)
    status_code = getattr(cause, "status_code", None)
    if status_code is not None:
        print(f"status_code: {status_code}", file=sys.stderr)
    body = getattr(cause, "body", None)
    if body is not None:
        print("error_body:", file=sys.stderr)
        try:
            print(json.dumps(body, ensure_ascii=False, indent=2), file=sys.stderr)
        except TypeError:
            print(body, file=sys.stderr)


def main() -> int:
    args = parse_args()
    loaded_env = load_env_file(args.env_file)
    config = get_config(args)

    print("API test configuration:")
    print(f"  env_file: {args.env_file}")
    print(f"  env_loaded: {'yes' if loaded_env else 'no'}")
    print(f"  base_url: {config['base_url'] or '<default>'}")
    print(f"  model: {config['model'] or '<missing>'}")
    print(f"  api_key: {mask_secret(config['api_key'])}")
    print(f"  retries: {config['retries']}")
    print(f"  timeout: {config['timeout']}")

    missing = [name for name in ("api_key", "model") if not config[name]]
    if missing:
        print(f"\nMissing required configuration: {', '.join(missing)}", file=sys.stderr)
        print("Set them in .env or pass --api-key/--model.", file=sys.stderr)
        return 2

    try:
        client = create_openai_client(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
        )
        response = chat_completion_with_retry(
            client=client,
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            retries=config["retries"],
            retry_delay=config["retry_delay"],
            temperature=config["temperature"],
            max_completion_tokens=config["max_completion_tokens"],
            return_response=True,
        )
    except Exception as exc:
        print_error(exc)
        return 1

    text = extract_message_text(response)
    print("\nRequest succeeded.")
    print(f"response_id: {getattr(response, 'id', '<unknown>')}")
    print(f"response_model: {getattr(response, 'model', '<unknown>')}")
    print(f"content: {text!r}")

    if config["dump_response"]:
        print("\nRaw response:")
        try:
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
        except Exception:
            print(repr(response))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
