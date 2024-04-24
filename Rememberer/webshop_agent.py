# Copyright 2023 SJTU X-Lance Lab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by Danyang Zhang @X-Lance.

import abc
import logging

from typing import List, Tuple
from typing import Callable, Optional
import agent_protos

import vh_to_html
import history
import numpy as np
import tiktoken
import itertools

logger = logging.getLogger("webshop")

Key = Tuple[str, str, str] # (observation, task, available_actions)
Action = Tuple[str, str] # (action, reason)

def remove_instruction(obs: str) -> List[str]:
    obs = obs.split("\n")
    obs = [x.strip() for x in obs]
    obs = [x for x in obs if not x.startswith("[button]") or x.startswith("[button] B0")]  # remove all buttons except the product code ones
    if "Instruction:" in obs:
        ins_idx = obs.index("Instruction:")
        return obs[:ins_idx] + obs[ins_idx+2:]
    else:   # mturk end screen, every page besides that should have instruction:
        return obs

class Agent(abc.ABC):
    #  class Agent {{{ # 
    def __init__(self, env_mode: str):
        #  method __init__ {{{ # 
        self._action_history: List[Action] = []
        self._env_mode: str = env_mode

        self._preprocess_observation: Callable[[str], List[str]]
        if env_mode=="html":
            self._preprocess_observation = vh_to_html.simplify_html
        elif env_mode=="text":
            self._preprocess_observation = vh_to_html.convert_simple_page
        elif env_mode=="text_rich":
            self._preprocess_observation = remove_instruction
        elif env_mode=="url":
            self._preprocess_observation = lambda url: [url]
        #  }}} method __init__ # 

    def reset(self):
        self._action_history.clear()
    def end( self
           , task: str
           , observation: str
           , reward: float
           , total_reward: float
           , available_actions: List[str]
           ):
        pass

    def __call__( self
                , task: str
                , observation: str
                , available_actions: List[str]
                ) -> str:
        #  method __call__ {{{ # 
        """
        Args:
            task (str): task instruction
            observation (str): observation
            reward (float): the last reward
            total_reward (float): the total history reward
            available_actions (List[str]): available_actions on the current observation

        Returns:
            Action: the action to take
        """

        action_tuple: Action = self._get_action( task
                                             , self._preprocess_observation(observation)
                                             , available_actions
                                             )
        action_str: str = action_tuple[0]
        action_reason: str = action_tuple[1]

        if action_str!="NOTHINGG":
            self._action_history.append(action_tuple)
        return action_str, action_reason
        #  }}} method __call__ # 

    @abc.abstractmethod
    def _get_action( self
                   , task: str
                   , observation: str
                   , reward: float
                   , total_reward: float
                   , available_actions: List[str]
                   ) -> Action:
        raise NotImplementedError()

    def train(self, train: bool):
        pass
    #  }}} class Agent # 

class ManualAgent(Agent):
    #  class ManualAgent {{{ # 
    def __init__(self, env_mode: str):
        super(ManualAgent, self).__init__(env_mode)

    def _get_action( self
                   , task: str
                   , observation: str
                   , reward: float
                   , total_reward: float
                   , available_actions: List[str]
                   ) -> Action:
        #  method _get_action {{{ # 
        print("Task:")
        print(task)
        print("Observation:")
        print("\n".join(observation))
        print("Action History:")
        print("\n".join(self._action_history))
        print("Last Reward:")
        print("{:.1f}".format(reward))
        print("Total Reward:")
        print("{:.1f}".format(total_reward))
        print("Available Action:")
        print(", ".join(available_actions))

        action_str: str = input("Please input the next action:")
        return action_str, "something"
        #  }}} method _get_action # 
    #  }}} class ManualAgent # 

class AutoAgent( Agent
               , agent_protos.OpenAIClient[Action]
               , agent_protos.HistoryReplayClient[Key, Action]
               ):
    #  class AutoAgent {{{ # 
    def __init__( self
                , history_replay: history.HistoryReplay[Key, Action]
                , filtered_history_replay: history.FilteredHistoryReplay
                , prompt_templates: agent_protos.TemplateGroup
                , api_key: str
                , model: str = "gpt-3.5-turbo-instruct"
                , max_tokens: int = 20
                , temperature: float = 0.1
                , stop: Optional[str] = None
                , request_timeout: float = 5.
                , static: bool = False
                , manual: bool = False
                , train: bool = True
                , env_mode: str = "text_rich"
                , norandom: bool = False
                ):
        #  method __init__ {{{ # 
        super(AutoAgent, self).__init__(env_mode)

        self._config_temperature: float = temperature
        #temperature = self._config_temperature if train else 0.
        super(Agent, self).__init__( prompt_templates
                                   , api_key
                                   , model
                                   , max_tokens
                                   , temperature
                                   , stop
                                   , request_timeout
                                   , 3.1
                                   , manual
                                   )

        self._input_length_limit: int = 3700

        self._tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(model)
        super(agent_protos.OpenAIClient, self).__init__( history_replay
                                                       , filtered_history_replay
                                                       , train
                                                       , self._tokenizer
                                                       , norandom
                                                       )

        self._static: bool = static
        #  }}} method __init__ # 

    def reset(self):
        super(AutoAgent, self).reset()
        #self._history_replay.new_trajectory()
    def end( self
           , task: str
           , observation: str
           , reward: float
           , total_reward: float
           , available_actions: List[str]
           ):
        #  method end {{{ # 
        self._history_replay.reset()
        
        # observation: str = "\n".join(self._preprocess_observation(observation))
        # available_actions: str = "\n".join(available_actions)
        # if self._train:
        #     last_action: Optional[Action] = self._action_history[-1]\
        #                                     if len(self._action_history)>0\
        #                                   else None
        #     self._history_replay.update( (observation, task, available_actions)
        #                                , reward
        #                                , last_action
        #                                , last_step=True
        #                                )
        #  }}} method end # 

    def _instantiate_input_template( self
                                   , task: str
                                   , observation: str
                                   , action_history: List[Action]
                                   , available_actions: str
                                   ):
        #  method _instantiate_input_template {{{ # 
        return self._prompt_templates.input_template.safe_substitute(
                                                        task=task
                                                      , instruction=task
                                                      , observation=\
                                                              "\n".join(
                                                                  map( lambda l: "  " + l
                                                                     , observation.splitlines()
                                                                     )
                                                                )
                                                      , actions=\
                                                              "\n".join(
                                                                  map( lambda act: "- " + act
                                                                     , map( " ".join
                                                                          , action_history[-min(5, len(action_history)):]
                                                                          )
                                                                     )
                                                                )
                                                      , available_actions=\
                                                              "\n".join(
                                                                  map( lambda act: "  <action>" + act + "</action>"
                                                                     , available_actions.splitlines()
                                                                     )
                                                                )
                                                      )
        #  }}} method _instantiate_input_template # 

    def _random_action(self, key: Key, encourages: bool = False) -> Action:
        #  method _random_action {{{ # 
        available_actions: List[str] = key[-1].splitlines()
        action: np.int64 = self._rng.integers(len(available_actions))
        if encourages:
            if available_actions[action]=="search":
                action_str: str = "search[{:}]".format(key[1])
                reason: str = ""
            else:
                action_str: str = "click[{:}]".format(available_actions[action])
                if available_actions[action]=="< prev":
                    reason: str = "The current item doesn't offer the desired options and I need to go back to check other items."
                elif available_actions[action]=="back to search":
                    reason: str = "The current item doesn't offer the desired options and I need to search for other items."
                elif available_actions[action]=="buy now":
                    reason: str = "All the options are ready now and I will click \"buy now\" to complete the shopping."
                else:
                    reason: str = "{:} conforms to the instruction.".format(available_actions[action])
        else:
            action_str: str = "click[{:}]".format(available_actions[action])
            if available_actions[action]=="search":
                reason: str = "The search button shouldn't be clicked."
            elif available_actions[action]=="features":
                reason: str = "There is no need to check the features."
            elif available_actions[action]=="description":
                reason: str = "There is no need to check the description."
            elif available_actions[action]=="reviews":
                reason: str = "There is no need to review."
            elif available_actions[action]=="buy now":
                reason: str = "Not all the requirements are ready now."
            elif available_actions[action]=="< prev":
                reason: str = "The current item offers the desired options and I don't need to go back to check other items."
            elif available_actions[action]=="back to search":
                reason: str = "The current item offers the desired options and I don't need to search for other items."
            else:
                reason: str = "{:} is not the desired item.".format(available_actions[action])
        return (action_str, reason)
        #  }}} method _random_action # 

    # def _action_to_string(self, action: Action, value: float) -> str:
    #     return "{:} -> {:.1f} {:}".format(action[0], value, action[1])
    
    # ablate the importance of the q values
    def _action_to_string(self, action: Action, value: float) -> str:
        return "{:} -> {:}".format(action[0], action[1])

    def _examplar_to_string( self
                           , index: int
                           , key: Key
                           , info_dict: history.HistoryReplay.InfoDict[Action]
                           , encouraged: str
                           ) -> str:
        #  method _examplar_to_string {{{ # original
        examplar: str = "Example {:d}:\n\n".format(index+1)\
                      + self._instantiate_input_template( task=key[1]
                                                        , observation=key[0]
                                                        , action_history=info_dict["action_history"]
                                                        , available_actions=key[2]
                                                        )\
                      + "\n"\
                      + self._prompt_templates.advice_template.safe_substitute(
                                                                encouraged=encouraged
                                                              )
                    # remove discouraged
                    #   + self._prompt_templates.advice_template.safe_substitute(
                    #                                             encouraged=encouraged
                    #                                           , discouraged=discouraged
                    #                                           )
        return examplar
        #  }}} method _examplar_to_string # 
    
    def _myrecord_to_string(self, idx: int, rec: history.MyRecord):
        examplar: str = f"Example {idx}\n\n" \
            + self._instantiate_input_template( task=rec.ins
                                              , observation=rec.obs
                                              , action_history=rec.past_actions     # idk i didn't implement this yet
                                              , available_actions='\n'.join(rec.avail_actions))\
            + "\n"\
            + self._prompt_templates.advice_template.safe_substitute(encouraged=self._action_to_string((rec.sugg_action, rec.reason), 0.0))
        return examplar

    def _parse_action(self, response: str) -> Action:
        #  method _parse_action {{{ # 
        return agent_protos.parse_action_with_optional(response)
        #  }}} method _parse_action # 

    def _get_action( self
                   , task: str
                   , observation: List[str]
                   , available_actions: List[str]
                   ) -> Action:
        #  method _get_action {{{ # 
        observation: str = "\n".join(observation)
        available_actions: str = "\n".join(available_actions)

        #  Construct New Input {{{ # 
        new_input: str = self._instantiate_input_template( task=task
                                                         , observation=observation
                                                         , action_history=self._action_history
                                                         , available_actions=available_actions
                                                         )
        nb_new_input_tokens: int = len(self._tokenizer.encode(new_input))
        example_tokens_limit: int = self._input_length_limit - nb_new_input_tokens
        #  }}} Construct New Input # 

        #  Construct Examplars {{{ # 
        if self._static:
            examplars: List[str] = [ (f"Example {i}:\n" + c) for i, c in enumerate(self._prompt_templates.canonicals)]
        else:
            examplars: List[str] = self._get_examplars( (observation, task, available_actions)
                                                      , example_tokens_limit
                                                      , 2
                                                      )

        example_str: str = "\n".join(reversed(examplars)).strip()
        #  }}} Construct Examplars # 

        prompt: str = self._prompt_templates.whole_template.safe_substitute( examples=example_str
                                                                           , new_input=new_input
                                                                           )
        action: Optional[Action] = self._get_response(prompt)
        if action is None:
            action_text: str = "NOTHINGG"
            reason: str = ""
        else:
            action_text: str
            reason: str
            action_text, reason = action

        logger.debug("Action: %s %s", action_text, reason)
        return (action_text, reason)
        #  }}} method _get_action # 

    def _update_history(self, task_idx, task, observation, available_actions, taken_action, reason, reward, done):
        #  Replay Updating {{{ # 
        if self._train:
            self._history_replay.update((observation, task, available_actions), reward, (taken_action, reason))
            self._filtered_history_replay.update(task_idx
                                       , (observation, task, available_actions)
                                       , reward
                                       , taken_action
                                       , reason
                                       , done
                                       )
        #  }}} Replay Updating # 

    def train(self, train: bool):
        super(agent_protos.OpenAIClient, self).train(train)
        #self._temperature = self._config_temperature if self._train else 0.
    #  }}} class AutoAgent # 
