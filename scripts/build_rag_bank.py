"""Root-AI curated terms from previously approved training material only."""
import json
from pathlib import Path
from witrans_tools.common import read_jsonl, write_json, now
from witrans_tools.rag import digest, OpenAIEmbeddings, EMBEDDING_MODEL, DIMENSIONS

# source family | English term | Chinese equivalent | sense anchors (either language)
# No confirmation/release content or development references are read by this builder.
TERMS = """
v2-train-075|eigenvalue|特征值|
v2-train-072|encoder|编码器|
v2-train-072|classification head|分类头|
v2-train-092|equivalence point|等当点|
v2-train-067|initialized|初始化|variable,declared,变量,声明
v2-train-076|necessary condition|必要条件|
v2-train-080|causal relationship|因果关系|
v2-train-081|response rate|回复率|sample,survey,样本,调查
v2-train-082|median|中位数|
v2-train-097|nominal wages|名义工资|
v2-train-097|purchasing power|购买力|
v2-train-099|opportunity cost|机会成本|
v2-train-074|normalization parameters|归一化参数|
v2-train-085|displacement|位移|force,work,力,做功
v2-train-087|steady current|稳恒电流|
v2-train-088|volume|体积|pressure,constant,压力,保持
v2-train-090|precipitate|沉淀|solution,dissolved,溶液,溶解
v2-train-091|reaction rate|反应速率|
v2-train-091|yield|产率|reaction,product,反应,产物
v2-train-093|control group|对照组|
v2-train-094|protein product|蛋白质产物|
v2-train-103|boundary condition|边界条件|
v2-train-106|differentiable|可微|
v2-train-108|treatment effect|处理效应|participants,groups,参与者,组间
v3-train-020|capacitor|电容器|
v3-train-021|dependent variable|因变量|
v3-train-021|independent variable|自变量|
v3-train-023|weight|权重|observation,numerical,观测,数值
v3-train-024|purity|纯度|
v3-train-025|milliamperes|毫安|
v3-train-026|sampling error|抽样误差|
v3-train-026|systematic bias|系统性偏差|
v3-train-030|bond|化学键|chemical,atoms,化学,原子
v4-mining-014|pressure difference|压差|
v4-mining-015|confidence interval|置信区间|
v4-mining-016|catalyst|催化剂|
v4-mining-016|activation energy|活化能|
v4-mining-016|equilibrium constant|平衡常数|
v4-mining-017|bank|河岸|erosion,stream,river,侵蚀,溪流,河流
v4-mining-018|bank|银行|financial,staffing,accounts,金融,人手,账户
v4-mining-019|scale|秤|weighing,flour,称重,面粉
v4-mining-020|scale|量表|psychology,questionnaire,心理,问卷
v4-mining-021|charge|费用|fee,luggage,费用,行李
v3-train-020|charge|电荷|capacitor,electric,电容,电学
v4-mining-023|pitch|音高|melody,singer,music,旋律,歌手,音乐
v4-mining-024|pitch|螺距|thread,machined,screw,螺纹,加工,螺钉
v5-seed-013|buffer|缓冲液|pH,acid,chemistry,酸,化学
v5-seed-014|measurement drift|测量漂移|
v5-seed-016|histogram|直方图|
v5-seed-019|cast|石膏|wrist,healed,plaster,手腕,愈合,石膏
v5-seed-020|cast|演员阵容|actors,theatrical,演员,戏剧
v5-seed-021|stress|应力|mechanical,steel,rod,力学,钢,杆
v5-seed-022|stress|压力|psychological,work,心理,工作
v5-seed-023|culture|培养物|bacterial,laboratory,sterile,细菌,实验室,无菌
v5-seed-024|culture|文化|social,organization,社会,组织
v5-seed-025|interest|利息|savings,account,compounding,储蓄,账户,复利
v5-seed-026|interest|兴趣|astronomy,students,history,天文,学生,历史
v5-seed-027|yield|收成|wheat,harvest,小麦,收获
v5-seed-028|yield|收益率|investment,financial,投资,金融
v5-seed-029|field|田地|planted,irrigation,种植,灌溉
v5-seed-030|field|字段|database,record,data type,数据库,记录,数据类型
v5-seed-031|induction|入职培训|employee,company,员工,公司
v5-seed-032|induction|归纳证明|mathematical,proof,theorem,数学,证明,定理
v6-academic-01|sample mean|样本均值|
v6-academic-01|population mean|总体均值|
v6-academic-03|phase change|相变|
v6-academic-04|cell membrane|细胞膜|
v6-academic-05|deposition|沉积|sediment,land,沉积物,土地
v6-academic-06|queue|队列|stack,back,front,栈,队尾,队首
v6-academic-06|stack|栈|queue,队列
v7-context-001|branch|分行|bank,customer,银行,顾客
v7-context-002|branch|分支|Git,developer,开发者
v7-context-003|channel|水道|boats,water,船,水
v7-context-004|channel|频道|chat,online,聊天,线上
v7-context-005|joint|接头|pipe,threaded,水管,螺纹
v7-context-006|joint|关节|knee,膝盖
v7-context-007|bass|海鲈鱼|fish,stall,鱼,摊
v7-context-008|bass|低音吉他|guitar,rehearsal,吉他,排练
v7-context-010|sole|鳎鱼|flatfish,fish,比目鱼,鱼
v7-context-011|terminal|航站楼|airport,passenger,机场,旅客
v7-context-012|terminal|终端|computer,commands,电脑,命令
v7-context-017|mole|鼹鼠|garden,burrowing,花园,穴居
v7-context-018|mole|卧底|spy,infiltrates,间谍,潜入
v7-context-019|bark|树皮|timber,tree,木材,树
v7-context-020|bark|吠叫声|dog,audio,狗,录音
v2-train-031|platform|站台|train,express,火车,快车
v2-train-033|gate|登机口|flight,airport,航班,机场
v2-train-037|shuttle|接驳车|terminal,airport,航站楼,机场
v2-train-041|adjoining rooms|相邻的房间|
v2-train-044|luggage storage|行李寄存|
v2-train-051|gravy|肉汁|
v2-train-053|main course|主菜|
v2-train-055|sparkling water|气泡水|
v2-train-055|still water|无气泡水|
v2-train-056|serving spoon|公勺|
v2-train-058|stock|高汤|dish,shellfish,foam,cooking,菜肴,贝类,泡沫,烹饪
v2-train-059|chili sauce|辣椒酱|
v2-train-061|service charge|服务费|
v3-train-005|hinge|铰链|
v3-train-009|waiting room|候车室|passengers,tickets,乘客,车票
v3-train-010|cable car|缆车|
v3-train-011|concourse|大厅|platform,station,站台,车站
v3-train-014|custard|蛋奶沙司|
v3-train-017|mustard|芥末|
v3-train-019|saucepan|汤锅|
v3-train-019|frying pan|煎锅|
v4-mining-005|tram|有轨电车|
v4-mining-007|boarding passes|登机牌|
v4-mining-008|hostel|青年旅舍|
v4-mining-009|slotted spoon|漏勺|
v4-mining-010|fold|翻拌|egg whites,batter,蛋白,面糊
v4-mining-010|batter|面糊|
v4-mining-012|anchovies|凤尾鱼|
v5-seed-001|radiator|散热器|
v5-seed-005|funicular|地面缆车|
v5-seed-008|accessible entrance|无障碍入口|
v5-seed-009|skim|撇去|foam,stock,泡沫,高汤
v5-seed-011|tahini|芝麻酱|
v5-seed-012|crimp|压合|edges,fork,pastry,边缘,叉子,酥皮
v6-daily-02|rake|耙子|
v6-travel-01|stroller|婴儿车|
v6-food-02|spatula|铲子|
v6-food-03|knead|揉|dough,面团
v6-food-08|parchment paper|烘焙纸|biscuits,baker,饼干,面包师
v10-daily-01|window latch|窗锁|
v10-daily-01|tenant|房客|
v10-daily-01|landlord|房东|
v10-daily-05|glazed pottery|上釉陶器|
v10-daily-07|external drive|外接存储设备|
v10-daily-08|noticeboard|公告板|
v10-daily-09|strap|表带|watch,手表
v10-daily-10|walking stick|拐杖|
v10-daily-13|measuring tape|卷尺|
v10-daily-13|caliper|卡尺|
v10-travel-02|audio guide|语音导览器|
v10-travel-02|admission fee|门票费用|
v10-travel-08|tent pitch|帐篷营位|
v10-travel-11|late checkout|延迟退房|
v10-travel-13|trailhead|步道入口|
v10-travel-16|breakfast voucher|早餐券|
v10-travel-18|deposit refund|押金退款|
v10-travel-19|wristband|腕带|
v10-food-01|chickpeas|鹰嘴豆|
v10-food-01|green peas|青豆|
v10-food-06|breadcrumbs|面包屑|
v10-food-08|olives|橄榄|
v10-food-10|cumin|孜然|
v10-food-10|coriander|香菜|ingredients,garnishes,食材,配料
v10-food-17|parsley|欧芹|
v10-food-17|basil|罗勒|
v10-food-18|brandy|白兰地|
v10-food-19|cross-contact|交叉接触|
v10-food-19|walnut sauce|核桃酱|
v10-food-20|canned beans|罐装豆|
v10-academic-01|missing values|缺失值|
v10-academic-02|default value|默认值|
v10-academic-06|structural bias|结构偏差|
v10-academic-08|counterexample|反例|
v10-academic-08|sufficiency|充分性|
v10-academic-13|solute|溶质|
v10-academic-15|merge sort|归并排序|
v10-academic-16|random seed|随机种子|
v10-academic-18|directed edge|有向边|
v10-academic-19|estimator|估计量|
v10-hard-01|a long shot|成功的希望渺茫|
v10-hard-02|on the fence|犹豫不决|
v10-hard-03|learning the ropes|学习业务|
v10-hard-04|with a grain of salt|以怀疑态度|
v10-hard-06|hit the nail on the head|一针见血|
v10-hard-07|passing the buck|推卸责任|
v10-hard-10|back to square one|从头再来|
v10-hard-11|red herring|转移注意力的干扰|
v10-hard-12|brushed up on|复习|
v10-hard-15|keep an eye on|密切关注|
v10-hard-16|joined forces|联手合作|
v10-hard-17|bumped into|偶然遇到|
v10-hard-19|tied up|忙于|engagement,安排
"""


def main():
    root = Path("runs/takeover-20261002/rag")
    root.mkdir(parents=True, exist_ok=True)
    source_paths = [Path("data/prepared/qwen35-v1/sft.jsonl"),
                    Path("data/prepared/v10-source-family-checked/train.jsonl")]
    rows = [r for p in source_paths for r in read_jsonl(p)]
    bank = []
    for line in TERMS.strip().splitlines():
        family, en, zh, anchors = line.split("|")
        sources = [r for r in rows if r["id"].startswith(family + "-")]
        if not sources or any(not r["source"].get("training_allowed") or r.get("review", {}).get("status") != "approved" for r in sources):
            raise ValueError("Unapproved/non-training source: " + family)
        # Root AI reads English/Chinese pairs and authors concise terminology equivalents,
        # which can normalize the old reference (e.g. still water), with provenance retained.
        bank.append({"id": f"term-{len(bank)+1:03}", "en": en, "zh": zh,
                     "anchors": anchors.split(",") if anchors else [], "source_id": family,
                     "source_groups": sorted({r["group_id"] for r in sources}),
                     "source_hashes": [digest(r) for r in sources],
                     "license": sorted({r['source']['license'] for r in sources}), "reviewer": "Codex AI",
                     "human_review": False})
    bank_path = Path("data/rag-reviewed-training-terms-20261002.jsonl")
    bank_path.write_bytes(("\n".join(json.dumps(r, ensure_ascii=False) for r in bank)+"\n").encode())
    write_json(root/"protocol.json", {"at": now(), "bank_hash": digest(bank), "entries": len(bank),
        "source_files": {str(p):digest(read_jsonl(p)) for p in source_paths}, "source_hash": digest(rows),
        "source_scope": "Previously approved synthetic training data only; no test references or outputs",
        "review": "Root Codex AI read cited bilingual source pairs; terminology mappings and sense anchors authored before RAG outputs",
        "human_review": False, "embedding_model": EMBEDDING_MODEL, "dimensions": DIMENSIONS,
        "profiles": ["semantic_terms", "local_terms"], "semantic_cosine_floor": 0.25,
        "max_terms": 3, "lexical_full_term_required": True, "ambiguous_term_requires_positive_anchor": True,
        "conflicting_senses": "abstain", "explicit_user_glossary": "takes_precedence",
        "quality_sets": ["known200", "public116"], "thresholds_changed": False,
        "adoption": "No new major/critical/direction/format/EOS errors; full source review; measurable net quality gain; formal 24x3 end-to-end mean<=4s P95<=8s reserved<=6.5GiB",
        "test_isolation": "Do not read confirmation or release in this experiment; no new training; no release claim"})
    client = OpenAIEmbeddings(cache_path=".cache/rag/index-embeddings.json")
    try:
        texts = ["Translation terminology: " + r["en"] + " = " + r["zh"] + ". Applicable context: " + ", ".join(r["anchors"]) for r in bank]
        vectors = client.embed(texts)
        write_json(root/"index.json", {"model": EMBEDDING_MODEL, "dimensions": DIMENSIONS,
                                    "bank_hash": digest(bank), "vectors": vectors})
        write_json(root/"index-receipt.json", {"at": now(), "bank_hash": digest(bank),
            "index_hash": digest(json.loads((root/"index.json").read_text())), "embedding_events": client.events,
            "estimated_embedding_cost_usd": sum(e["input_tokens"] for e in client.events)*0.13/1_000_000})
        print({"entries": len(bank), "events": client.events})
    finally:
        client.close()


if __name__ == "__main__":
    main()
