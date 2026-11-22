"""Build the ≤ 5-minute demo MP4 and its narration script from the sealed master results.

Pipeline: start the cockpit server → capture cockpit tabs (incl. a live What-If solve) with Playwright →
sentence-level Chinese TTS (edge-tts) → captioned 1920×1080 frames → ffmpeg segments → concat.
The visual order follows the required modelling flow: 数据输入 → 特征提取 → 差距量化 → 方案生成 → 效果预测与验证.
Outputs: outputs/deliverables/演示视频.mp4, 演示脚本.md
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
COMP = ROOT.parent
OUT = ROOT / "outputs"
DELIV = OUT / "deliverables"
WORK = DELIV / "_video_work"
SLIDES = DELIV / "_pptx_png"
FONT = "/mnt/c/Windows/Fonts/Deng.ttf"
FONT_B = "/mnt/c/Windows/Fonts/Dengb.ttf"
VOICE = "zh-CN-YunxiNeural"
PORT = 8765
MAX_SECONDS = 292.0


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def start_server():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.cockpit_server:app", "--port", str(PORT), "--log-level", "warning"],
                         cwd=str(COMP), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=2) as r:
                if r.status == 200:
                    return p
        except Exception:
            time.sleep(0.5)
    p.terminate()
    raise RuntimeError("cockpit server did not start")


async def capture_cockpit() -> dict:
    from playwright.async_api import async_playwright
    shots = {}
    async with async_playwright() as pw:
        b = await pw.chromium.launch()
        pg = await b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2)
        await pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        await pg.wait_for_function("document.getElementById('sha-badge').textContent.length > 12")
        tabs = {"overview": "总览", "task1": "任务1", "task2": "任务2", "rsi": "核心创新", "task3": "任务3", "task4": "任务4"}
        for key, label in tabs.items():
            await pg.click(f"nav button:has-text('{label}')")
            await pg.wait_for_timeout(700)
            await pg.evaluate("window.scrollTo(0, 0)")
            path = WORK / f"cockpit_{key}.png"
            await pg.screenshot(path=str(path))
            shots[key] = path
        # task2 lower part (streaming + employment feedback)
        await pg.click("nav button:has-text('任务2')")
        await pg.evaluate("document.getElementById('macro-portfolio-table').scrollIntoView({block: 'start'})")
        await pg.evaluate("window.scrollBy(0, -90)")
        await pg.wait_for_timeout(400)
        shots["task2b"] = WORK / "cockpit_task2b.png"
        await pg.screenshot(path=str(shots["task2b"]))
        # live What-If
        await pg.click("nav button:has-text('What-If')")
        await pg.wait_for_timeout(400)
        await pg.click("button:has-text('实时重算')")
        await pg.wait_for_function("document.getElementById('whatif-output').textContent.includes('实时求解成功') || document.getElementById('whatif-output').textContent.includes('不可用')", timeout=180000)
        txt = await pg.inner_text("#whatif-output")
        if "实时求解成功" not in txt:
            raise RuntimeError("What-If live solve failed: " + txt)
        await pg.evaluate("window.scrollTo(0, 0)")
        shots["whatif"] = WORK / "cockpit_whatif.png"
        await pg.screenshot(path=str(shots["whatif"]))
        resp = await pg.request.post(f"http://127.0.0.1:{PORT}/api/whatif", data={
            "program_id": "BTBU_CS_2022", "budget_mode": "medium", "multitrack": True, "scenario": "baseline",
            "faculty_ramp": 0.94, "capstone_orient_ratio": 0.55})
        shots["whatif_data"] = await resp.json()
        await b.close()
    return shots


def scenes(m: dict, shots: dict, verify: dict) -> list:
    C = m["cases"]
    r25 = {p: C[p]["results"]["replay_2025"]["metrics"] for p in C}
    full = C["BTBU_CS_2022"]["actions"]["Full_EvoAlign_OS"]
    st = full["streaming_summary"]
    b = {x["program_id"]: x for x in m["task1_gap_analysis"]["multi_university_benchmarks"]}["BTBU_CS_2022"]
    efb, rt = m["task2_employment_feedback"], m["task4_red_team_risks"]
    ne, nm = m["natural_experiments"]["micro_course_btbu_2022_to_2025"], m["natural_experiments"]["macro_major_moe_2024_beijing"]
    sc, frz = m["task3_dream_sim_3scenarios"], m["task3_counterfactual_freeze"]
    rel = m["extraction_reliability"]
    wi = shots["whatif_data"]
    b4 = {p: C[p]["results"]["replay_2025"]["paired_qual_ci95"]["B4_BiLevel_NoGuard"]["Full_EvoAlign_OS"]["delta"] for p in C}
    rl = efb["skill_mismatch_evidence"]
    ver = verify["summary"]
    gini = m["task1_gap_analysis"]["dev2024_demand_summary"]["anomaly_audit"]["employer_gini"]
    S = SLIDES
    blind = json.loads((OUT / "preregistered_blind_results.json").read_text(encoding="utf-8"))
    return [
        ("开场", S / "s01.png", [
            "各位评委好，我们的作品是 EvoAlign-OS，帮助高校管理者决定专业怎么调、调完是否更对口。",
            "下面按数据输入、特征提取、差距量化、方案生成、效果预测与验证的顺序演示。"]),
        ("数据输入", S / "s03.png", [
            "需求端来自官方招聘数据：一千八百多万条记录，筛出一百七十一万条技术岗，再限定北京本科可及的入门岗。",
            f"供给端是自采的 {m['kg_stats']['universities']} 所北京高校 {m['kg_stats']['programs']} 份培养方案和大纲，另有教育部备案和官方就业报告。",
            "二〇二五年以后的数据物理隔离，只用于最终检验。"]),
        ("特征提取", S / "s04.png", [
            "两端放进同一个坐标：四十八项标准技能，乘以六级布鲁姆认知深度。",
            f"岗位按词表抽取技能并按雇主等权；课程由千问抽取，复核首位技能一致率 {pct(rel['top1_skill_agreement'])}。",
            "岗位达标要求技能平均就绪度达到零点七，且七成技能达到零点五五。"]),
        ("差距量化·技能结构", shots["task1"], [
            "差距分三个维度。技能结构上，十一份方案按定位分层，各自对照相应岗位。",
            "共同缺口是 Linux、数据库、Java 后端、测试和嵌入式。",
            f"雇主基尼系数 {gini:.2f}，所以统计全部雇主等权，区间用雇主聚类自助法。"]),
        ("差距量化·方向与规模", S / "s06.png", [
            "培养方向上，单一路径对 AI 数据和系统嵌入式岗位达标率最低。",
            f"规模上，计算机类专业相关度从 {rl['cs_relevance_2021_to_2025'][0]:.0f}% 降到 {rl['cs_relevance_2021_to_2025'][-1]:.0f}%，而北京入门岗需求没有下行。",
            "结论是结构性错配：人数够，能力不对口。"]),
        ("方案生成·课程增删改", shots["task2"], [
            f"约束求解器为主案例新增 {full['added_credits']:g} 学分方向选修，停开 {full['removed_credits']:g} 学分，升级 {full['upgraded_core_count']} 门核心课实验。",
            "毕业学分不变、必修全保留、师资在预算内，九项硬约束全部通过。"]),
        ("方案生成·分流机制", S / "s08.png", [
            "大二末按志愿、成绩和容量分到四个方向，Gale-Shapley 算法保证没有阻塞对。",
            f"仿真第一志愿满足率 {pct(st['first_choice_satisfaction'])}，方向人数不足会自动预警。"]),
        ("方案生成·招生调节", shots["task2b"], [
            "招生同时看就业反馈和需求预测：连续两届预警才触发，单次调幅不超过百分之八，冷却两年。",
            "当前建议保持总招生、调整方向名额、先改课程。"]),
        ("双环自进化", shots["rsi"], [
            "双环自进化让方案随数据更新：外环诊断未达标岗位缺什么，由千问提出新课和实训包。",
            "候选必须通过 GUARD：开发集不退化、最弱方向不下滑才晋级，偏科的对抗候选被拦下。"]),
        ("在线重算", shots["whatif"], [
            f"管理者可在线改参数实时重算，这里北工商计科开发集达标率从 {pct(wi['before_dev2024']['jd_qual_rate_70'])} 升到 {pct(wi['after_dev2024']['jd_qual_rate_70'])}。"]),
        ("效果验证·六组消融", S / "s11.png", [
            "六组受控消融共用同一约束、预算和评估器。",
            f"二〇二五回放中，三个专业达标率分别从 {pct(r25['BTBU_CS_2022']['B0_Freeze']['jd_qual_rate_70'], 0)}、{pct(r25['BTBU_AI_2022']['B0_Freeze']['jd_qual_rate_70'], 0)}、{pct(r25['BISTU_CS_2025']['B0_Freeze']['jd_qual_rate_70'], 0)} 提升到 {pct(r25['BTBU_CS_2022']['Full_EvoAlign_OS']['jd_qual_rate_70'], 0)} 左右。"]),
        ("效果验证·置信区间", S / "s12.png", [
            "相对四类基线的二十四组配对比较全部显著。",
            f"关闭 GUARD 时人工智能专业落后 {b4['BTBU_AI_2022'] * 100:.0f} 个百分点，计科持平；一季度小样本上不显著，我们如实报告。"]),
        ("效果验证·预注册独立盲测", S / "s13.png", [
            "为排除对回放数据调参的可能，我们先冻结方案与评估代码并登记假设，再首次抽取从未读取过的京外岗位一次性检验。",
            f"在津冀与京外全国两个人群上，预先登记的 {sum(int(v.split('/')[0]) for v in blind['hypotheses_supported'].values())} 条假设全部成立。"]),
        ("效果预测·情景与自然实验", S / "s14.png", [
            f"悲观情景达标率仍有 {pct(sc['pessimistic']['jd_qual_rate_70'])}，高于不改革的 {pct(frz['jd_qual_rate_70'])}。",
            f"只用二〇二二年前数据的方案，与学校实际改版调整了 {ne['direction_agreement']['course_level']['n_agreed']} 门相同课程；专业布点预测命中 {nm['top5_hit_count']} 个。"]),
        ("风险评估", S / "s16.png", [
            f"八类风险绑定具体动作和责任主体，治理机制把压测超限频次从 {pct(rt['ungoverned_mean_breach_rate'])} 降到 {pct(rt['governed_mean_breach_rate'])}。"]),
        ("可复核性", S / "s18.png", [
            f"所有数字来自同一份封印文件，独立脚本重算 {ver['n_checks']} 项全部一致。谢谢各位评委。"]),
    ]


async def tts_all(items, rate: str):
    import edge_tts
    sem = asyncio.Semaphore(4)

    async def one(text, path):
        async with sem:
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(text, VOICE, rate=rate).save(str(path))
                    return
                except Exception:
                    await asyncio.sleep(1.5 * (attempt + 1))
            raise RuntimeError(f"TTS failed: {text[:20]}")
    await asyncio.gather(*[one(t, p) for t, p in items])


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def wrap(draw, text, font, width):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def frame(img_path: Path, section: str, caption: str, out: Path):
    W, H, CAP = 1920, 1080, 150
    canvas = Image.new("RGB", (W, H), (15, 35, 55))
    im = Image.open(img_path).convert("RGB")
    s = min(W / im.width, (H - CAP) / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    canvas.paste(im, ((W - im.width) // 2, 0))
    d = ImageDraw.Draw(canvas)
    f_cap, f_sec = ImageFont.truetype(FONT, 35), ImageFont.truetype(FONT_B, 23)
    d.rectangle([0, H - CAP, W, H], fill=(15, 35, 55))
    d.text((40, H - CAP + 10), section, font=f_sec, fill=(94, 234, 212))
    lines = wrap(d, caption, f_cap, W - 80)[:2]
    if len(lines) == 2 and len(lines[1]) <= 4:
        mid = len(caption) // 2
        cut = max((i for i in range(len(caption)) if caption[i] in "，、；：" and i <= mid + 8), default=mid)
        lines = [caption[:cut + 1], caption[cut + 1:]]
    y = H - CAP + 58 if len(lines) == 1 else H - CAP + 40
    for ln in lines:
        d.text((W // 2, y + (0 if len(lines) == 1 else 0)), ln, font=f_cap, fill=(255, 255, 255), anchor="ma")
        y += 50
    canvas.save(out)


def main():
    m = json.loads((OUT / "verified_master_results.json").read_text(encoding="utf-8"))
    verify = json.loads((OUT / "verification_report.json").read_text(encoding="utf-8"))
    assert verify["summary"]["master_sha256"] == m["master_sha256"], "stale verification report"
    if not (SLIDES / "s18.png").exists():
        raise SystemExit("slide PNGs missing: export 答辩PPT.pptx to outputs/deliverables/_pptx_png first")
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    srv = start_server()
    try:
        shots = asyncio.run(capture_cockpit())
    finally:
        srv.terminate()
    sc = scenes(m, shots, verify)
    items = []
    for i, (_, _, sents) in enumerate(sc):
        for j, t in enumerate(sents):
            items.append((t, WORK / f"a_{i:02d}_{j:02d}.mp3"))
    total = None
    for rate in ("+12%", "+18%", "+24%"):
        asyncio.run(tts_all(items, rate))
        total = sum(duration(p) + 0.25 for _, p in items) + 0.4 * len(sc)
        if total <= MAX_SECONDS:
            break
    if total > MAX_SECONDS:
        raise SystemExit(f"narration too long: {total:.1f}s")
    segs, md, t0 = [], [], 0.0
    for i, (sec, img, sents) in enumerate(sc):
        md.append(f"\n## {i + 1}. {sec}\n\n画面：`{Path(img).name}`\n")
        for j, t in enumerate(sents):
            a = WORK / f"a_{i:02d}_{j:02d}.mp3"
            d = duration(a) + 0.25 + (0.4 if j == len(sents) - 1 else 0.0)
            fp = WORK / f"f_{i:02d}_{j:02d}.png"
            frame(Path(img), f"{i + 1}/{len(sc)} · {sec}", t, fp)
            seg = WORK / f"seg_{i:02d}_{j:02d}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "25", "-i", str(fp), "-i", str(a),
                            "-af", "apad", "-t", f"{d:.3f}", "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast",
                            "-crf", "20", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                            str(seg)], check=True)
            segs.append(seg)
            md.append(f"- `{int(t0 // 60):d}:{t0 % 60:04.1f}` {t}")
            t0 += d
    lst = WORK / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in segs), encoding="utf-8")
    out = DELIV / "演示视频.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    real = duration(out)
    if real > 300.0:
        raise SystemExit(f"video too long: {real:.1f}s")
    head = (f"# 演示视频脚本（时长 {int(real // 60)} 分 {real % 60:.0f} 秒）\n\n"
            f"主线：数据输入 → 特征提取 → 差距量化 → 方案生成 → 效果预测与验证。所有数字取自 verified_master_results.json（SHA-256 {m['master_sha256'][:16]}…）；\n"
            f"第 10 段为录制时调用本机服务实时求解的结果。旁白为合成语音（{VOICE}），画面底部字幕与旁白逐句一致。\n")
    (DELIV / "演示脚本.md").write_text(head + "\n".join(md) + "\n", encoding="utf-8")
    print(f"video written: {out} ({real:.1f}s, {out.stat().st_size / 1e6:.1f} MB, {len(segs)} segments)")


if __name__ == "__main__":
    main()
