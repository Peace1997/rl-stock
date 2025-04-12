import os
from dataclasses import dataclass

from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm import tqdm


def make_dirs(*dirs):
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)


def clean_dirs(path: str):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            # remove files first
            for name in files:
                os.remove(os.path.join(root, name))
            # remove folders
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)


@dataclass
class Args:
    llm_config: dict
    topic_lst: list
    travily_api_key_path: str
    search_top_k: int
    output_path: str


PROMPT = """
### 你的角色：
你是一位专业的中文金融市场分析师，拥有多年从业经验，擅长根据海量信息提取有价值的资讯，并撰写具有专业深度和可读性的市场行情分析报告。

---

### 输入信息说明：
你将收到一组与当前金融话题相关的网络检索结果，格式为多条内容片段。这些信息可能包括新闻报道、研究分析、社交媒体讨论等。

变量说明：
- {doc}：为你提供的原始参考信息（已合并为一段长文本）
- {topic}：你要分析的金融话题或市场热点

---

### 任务目标：
请基于参考信息，围绕指定主题撰写一篇**中文行情分析报告**，需满足以下要求：

1. **甄别有效信息**：从参考内容中筛选与主题最相关的资讯，忽略无关内容。
2. **专业表达**：用金融专业人士的视角总结和分析，语言准确、逻辑清晰。
3. **格式规范**：使用标准 Markdown 格式撰写，确保内容结构良好、段落分明。
4. **避免引用来源**：不得直接提及原始信息的来源（如“据xxx网站”）。
5. **信息不足时处理**：如所有参考内容都无有效信息，应直接回复：“目前没有检索到相关资料，无法撰写。”

---

### 报告结构要求（严格按照以下结构撰写）：

#### 一句话摘要
用一句话简明总结“{topic}”当前的市场情况或整体趋势。

#### 第一段：主题相关热点信息
- 提炼{topic}相关的重要消息、事件或政策变动。
- 如果有多个方面的信息，请分类简述。

#### 第二段：市场行情走势分析
- 分析该主题在市场上的表现趋势、产业动向或资本反应。
- 可结合资金流向、股价变化、估值趋势等进行解释。

#### 第三段：总结与展望
- 简要总结当前形势；
- 提出你作为分析师对未来市场表现的预测或建议。

---

### 输出要求：
- 请用**中文**撰写；
- 请使用**Markdown 格式**输出，包括适当的标题（如“## 市场走势分析”）和段落；
- 内容应逻辑清晰，条理分明，文字简练但不丢失重要信息。

"""


class Reporter:
    def __init__(self, args: Args):
        self.args = args

        # 实现网络搜索工具
        with open(args.travily_api_key_path, "r") as file:
            api_key = file.read().replace("\n", "")

        os.environ["TAVILY_API_KEY"] = api_key
        self.web_search_tool = TavilySearchResults(k=args.search_top_k)

        # 大模型初始化
        llm_config = args.llm_config
        llm = ChatOpenAI(
            model=llm_config["llm_mdl_name"],
            openai_api_key=llm_config["llm_api_key"],
            openai_api_base=llm_config["llm_server_url"],
            max_tokens=llm_config["llm_max_tokens"],
            temperature=llm_config["llm_temperature"],
        )
        prompts_report = PromptTemplate(
            template=PROMPT,
            input_variables=["doc", "topic"],
        )
        self.writer_llm = prompts_report | llm | StrOutputParser()

    def generate_report(self):
        tot_num = len(self.args.topic_lst)
        for topic in tqdm(self.args.topic_lst, total=tot_num):
            docs = self.web_search_tool.invoke({"query": topic})
            # Debug: Print the type and content of docs
            print(f"Type of docs: {type(docs)}")
            print(f"Content of docs: {docs}")

            # Handle the data based on its actual structure
            if isinstance(docs, list):
                if all(isinstance(d, dict) and "content" in d for d in docs):
                    web_results = "\n---\n".join([d["content"] for d in docs])
                else:
                    # If docs is a list but elements aren't dictionaries with "content" key
                    web_results = "\n---\n".join([str(d) for d in docs])
            else:
                # If docs isn't a list at all
                web_results = str(docs)

            ret = self.writer_llm.invoke({"doc": web_results, "topic": topic})

            # write to output
            out_file_path = os.path.join(self.args.output_path, f"{topic}_report.md")
            with open(out_file_path, "w", encoding="utf-8") as f:
                f.write(ret)


def main():
    # Read API key from file
    api_key_file = os.path.join(os.path.dirname(__file__), "api_files/llm_api_key.txt")
    with open(api_key_file, "r") as f:
        api_key = f.read().strip()
    
    llm_config = {
        "llm_api_key": api_key,
        "llm_mdl_name": "DeepSeek-R1",  # 或 "gpt-3.5-turbo"
        "llm_server_url": "https://aihubmix.com/v1",  # OpenAI 官方地址
        "llm_max_tokens": 2048,
        "llm_temperature": 0.7,
    }

    # Using relative path for Tavily API key
    travily_api_key_file = os.path.join(os.path.dirname(__file__), "api_files/tavily_search_api.txt")
    search_top_k = 10
    # Modify the output_path in your script
    # Set output path as a subdirectory in the same location as this script
    output_path = os.path.join(os.path.dirname(__file__), "output")
    make_dirs(output_path)

    topic_lst = ["中美关税政策"]
    args = Args(
        llm_config=llm_config,
        topic_lst=topic_lst,
        travily_api_key_path=travily_api_key_file,
        search_top_k=search_top_k,
        output_path=output_path,
    )
    reporter = Reporter(args)
    reporter.generate_report()


if __name__ == "__main__":
    main()
