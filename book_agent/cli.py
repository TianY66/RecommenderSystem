"""Command-line entry points for the offline demo and reproducible evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .agent import BookAgent
from .agent_eval import load_cases, run_agent_eval
from .catalog import BookCatalog
from .data import DemoData, load_demo_data, temporal_leave_last_out
from .metrics import aggregate_ranking_metrics
from .providers import OpenAIResponsesProvider
from .ranking import RecommendationEngine
from .tools import BookTools

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CASES = DEFAULT_DATA_DIR / "agent_eval_cases.jsonl"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_recommenders(data: DemoData, *, ks: Sequence[int] = (10, 20)) -> dict:
    """Evaluate popularity and ItemCF on the same per-user temporal holdout."""

    split = temporal_leave_last_out(data.interactions)
    catalog = BookCatalog(data.items)
    recommender = RecommendationEngine(catalog, data.users).fit(split.train)
    users = sorted(split.test_targets)
    max_k = max(ks, default=0)
    popularity_rankings = {
        user_id: recommender.recommend_popularity(user_id, limit=max_k)
        for user_id in users
    }
    itemcf_rankings = {
        user_id: recommender.recommend_itemcf(user_id, limit=max_k)
        for user_id in users
    }
    relevant = {user_id: {item_id} for user_id, item_id in split.test_targets.items()}
    return {
        "data": {
            "catalog_items": len(data.items),
            "users": len(data.users),
            "interaction_events": len(data.interactions),
        },
        "split": {
            "strategy": "per-user temporal leave-last-distinct-item-out",
            "train_events": len(split.train),
            "test_users": len(split.test_targets),
            "skipped_users_with_fewer_than_two_items": split.skipped_user_count,
        },
        "models": {
            "popularity": aggregate_ranking_metrics(popularity_rankings, relevant, ks),
            "itemcf": aggregate_ranking_metrics(itemcf_rankings, relevant, ks),
        },
    }


def _components(data: DemoData) -> tuple[BookCatalog, RecommendationEngine, BookTools]:
    catalog = BookCatalog(data.items)
    recommender = RecommendationEngine(catalog, data.users).fit(data.interactions)
    return catalog, recommender, BookTools(catalog, recommender)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="book-agent")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Run the local search, ranking, and details flow")
    demo.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    demo.add_argument("--query", default="算法")
    demo.add_argument("--user-id")
    demo.add_argument("--category")
    demo.add_argument("--keyword")
    demo.add_argument("--min-price", type=float)
    demo.add_argument("--max-price", type=float)
    demo.add_argument("--limit", type=int, default=5)

    recommender = commands.add_parser(
        "evaluate-recommender", help="Run temporal Recall/NDCG/MRR evaluation"
    )
    recommender.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    recommender.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    recommender.add_argument(
        "--output", type=Path, default=DEFAULT_REPORTS_DIR / "recommender_eval.json"
    )

    agent_eval = commands.add_parser(
        "evaluate-agent", help="Run the JSONL cases against a real Responses API model"
    )
    agent_eval.add_argument("--provider", choices=("openai", "deepseek"), default="openai")
    agent_eval.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    agent_eval.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    agent_eval.add_argument(
        "--output", type=Path, default=DEFAULT_REPORTS_DIR / "agent_eval.json"
    )

    ask = commands.add_parser(
        "ask", help="Interactively ask the book agent a question using a real model"
    )
    ask.add_argument("--provider", choices=("openai", "deepseek"), default="deepseek")
    ask.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ask.add_argument("--query", help="Ask one question and exit; omit to start an interactive session")
    return parser


def _run_demo(args: argparse.Namespace) -> dict:
    data = load_demo_data(args.data_dir)
    catalog, _, tools = _components(data)
    session = tools.new_session()
    search = session.call(
        "search_catalog",
        {
            "query": args.query,
            "category": args.category,
            "keyword": args.keyword,
            "min_price": args.min_price,
            "max_price": args.max_price,
            "limit": args.limit,
        },
    )
    candidate_ids = [row["item_id"] for row in search["candidates"]]
    ranked = None
    if args.user_id and candidate_ids:
        ranked = session.call(
            "rank_candidates_for_user",
            {
                "user_id": args.user_id,
                "query": args.query,
                "candidate_item_ids": candidate_ids,
                "limit": min(args.limit, 20),
            },
        )
        selected_ids = [row["item_id"] for row in ranked["ranked_candidates"]]
    else:
        selected_ids = candidate_ids

    details = session.call("get_book_details", {"item_ids": selected_ids[: min(args.limit, 20)]})
    return {
        "query": args.query,
        "user_id": args.user_id,
        "search_count": search["count"],
        "ranked_candidates": ranked,
        "verified_books": details["items"],
    }


def _load_environment_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = create_parser()
    args = parser.parse_args(argv)
    _load_environment_file()

    if args.command == "demo":
        result = _run_demo(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-recommender":
        if any(k <= 0 for k in args.ks):
            parser.error("--ks values must all be positive")
        report = evaluate_recommenders(load_demo_data(args.data_dir), ks=tuple(args.ks))
        _write_json(args.output, report)
        print(json.dumps({"report": str(args.output), **report}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-agent":
        try:
            provider = OpenAIResponsesProvider.from_env(provider=args.provider)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        data = load_demo_data(args.data_dir)
        catalog, _, tools = _components(data)
        agent = BookAgent(tools, provider, model_name=provider.model_name)
        report = run_agent_eval(
            agent,
            load_cases(args.cases),
            catalog,
            output_path=args.output,
        )
        print(
            json.dumps(
                {"report": str(args.output), "model": provider.model_name, **report["summary"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "ask":
        try:
            provider = OpenAIResponsesProvider.from_env(provider=args.provider)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        data = load_demo_data(args.data_dir)
        catalog, _, tools = _components(data)
        agent = BookAgent(tools, provider, model_name=provider.model_name)
        return _run_ask(agent, args.query)

    parser.error("unknown command")
    return 2


def _write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_ask(agent: BookAgent, query: str | None) -> int:
    if query is not None:
        question = query.strip()
        if not question:
            print("问题不能为空。")
            return 2
        result = agent.run(question)
        _print_agent_result(result)
        return 0 if result.success else 1

    print("图书推荐 Agent 已就绪。每条问题独立处理；输入 q、exit 或 退出结束。")
    while True:
        try:
            question = input("\n你：").strip()
        except EOFError:
            print("\n会话结束。")
            return 0
        except KeyboardInterrupt:
            print("\n会话结束。")
            return 0

        if question.casefold() in {"q", "quit", "exit", "退出"}:
            print("会话结束。")
            return 0
        if not question:
            continue

        result = agent.run(question)
        _print_agent_result(result)


def _print_agent_result(result: Any) -> None:
    print(f"\nAgent：{result.answer}")
    if result.error:
        print(f"状态：未完成（{result.error}）")
    else:
        print("状态：完成")

    labels = {
        "search_catalog": "目录检索",
        "rank_candidates_for_user": "偏好排序",
        "get_book_details": "详情核验",
    }
    stages = []
    for call in result.trace.get("tool_calls", []):
        label = labels.get(call.get("name"), call.get("name", "未知工具"))
        if call.get("error"):
            stages.append(f"{label}失败")
        else:
            count = len(call.get("result_item_ids", []))
            stages.append(f"{label}（{count} 本）")
    print("工具轨迹：" + (" → ".join(stages) if stages else "无需调用工具"))

    rank_calls = [
        call for call in result.trace.get("tool_calls", [])
        if call.get("name") == "rank_candidates_for_user"
    ]
    if rank_calls:
        context = rank_calls[-1].get("user_context", {})
        if context.get("personalization_applied"):
            print("个性化：候选命中了用户历史或画像信号。")
        elif context.get("known_user"):
            print("个性化：该用户有历史或画像，但当前候选没有匹配信号。")
        else:
            print("个性化：没有可用的用户历史或画像，按检索相关度排序。")

    if result.trace.get("answer_fallback_used"):
        fallback_labels = {
            "no_candidates": "筛选后没有候选，已说明无结果",
            "unsupported_answer_facts": "模型回答包含目录无法证实的内容，已改用核验字段",
            "unverified_item_id": "模型引用了未核验书目，已改用核验字段",
            "title_mismatch": "模型书名与书目编号不匹配，已改用核验字段",
            "missing_citation": "模型回答缺少书目引用，已补为核验候选",
            "recommendation_count_mismatch": "模型返回数量不符，已按要求调整",
        }
        reason = result.trace.get("answer_fallback_reason")
        print("回答保护：" + fallback_labels.get(reason, "采用了基于已核验目录详情的安全回答"))
    duration_ms = result.trace.get("duration_ms")
    if duration_ms is not None:
        print(f"耗时：{duration_ms:.0f} ms")


if __name__ == "__main__":
    raise SystemExit(main())
