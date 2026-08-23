import { aiProcessText, ChatMessage } from "./ai";
import { broadcast } from "./ws";

export class Segment {
  start: number;
  end: number;
  text: string;
  id: string;
  opti_text: string;
  en_text: string;

  constructor(
    id: string = "",
    start: number = 0.0,
    end: number = 0.0,
    text: string = "",
    opti_text: string = "",
    en_text: string = ""
  ) {
    this.id = id;
    this.start = start;
    this.end = end;
    this.text = text;
    this.opti_text = opti_text;
    this.en_text = en_text;
  }

  static fromObject(obj: any): Segment {
    return new Segment(
      obj.id,
      obj.start,
      obj.end,
      obj.text,
      obj.opti_text,
      obj.en_text
    );
  }

  static fromJSONString(jsonString: string): Segment {
    return Segment.fromObject(JSON.parse(jsonString));
  }

  async previewInput(): Promise<void> {
    // 共享对话流 + 队列积压原文作补充上文 + 仅英译开关
    try {
      const { results } = await aiProcessText({
        history: conversationHistory,
        pendingContext: pendingQueue.slice(-3),
        texts: [this.text],
        enOnly: true,
      });
      this.en_text = results[0].translated;
    } catch {
      this.en_text = this.text; // 失败回退原文,正式翻译稍后覆盖
    }
  }
  

  
  toJSONString() {
    return JSON.stringify(this);
  }
}

let currentSegment: Segment = new Segment();
let confirmedSegments: Segment[] = [];

// 共享对话流: confirmed 每批成功后追加一轮(user原文+模型原始返回,append后不变),
// preview 与 confirmed 打同一条不断增长的前缀以最大化缓存命中
let conversationHistory: ChatMessage[] = [];

export function saveCurrentSegment(s: Segment): void {
  currentSegment = s;
}

export function getCurrentSegment(): Segment {
  return currentSegment;
}

export function getConfirmedSegments(): Segment[] {
  return confirmedSegments;
}

export function saveConfirmedSegment(seg: Segment): Segment[] {
  let id = confirmedSegments.length;
  seg.id = id.toString();
  let contextSegments = confirmedSegments.slice();
  confirmedSegments.push(seg);
  return contextSegments;
}

// ===== confirmed 串行批处理队列 =====
// 串行消费保证下一批处理时历史条目均为翻译完成态(消除竞态,稳定缓存前缀);
// 队列积压时合并为批量调用,每批最多 confirmedBatchMax 条,超出部分循环补齐
const runtimeConfig = useRuntimeConfig();
const CONFIRMED_BATCH_MAX = runtimeConfig.confirmedBatchMax || 2;
const HISTORY_MAX_ROUNDS = runtimeConfig.historyMaxRounds || 20;
const pendingQueue: Segment[] = [];
let draining = false;

// 入队时即赋唯一id:前端按id匹配confirmed→update替换,广播confirmed时id必须已就绪
let nextSegmentId = 0;

export function enqueueConfirmed(seg: Segment): void {
  seg.id = nextSegmentId.toString();
  nextSegmentId++;
  pendingQueue.push(seg);
  if (!draining) {
    draining = true;
    void drainQueue();
  }
}

async function drainQueue(): Promise<void> {
  try {
    while (pendingQueue.length > 0) {
      const batch = pendingQueue.splice(0, CONFIRMED_BATCH_MAX);
      batch.forEach(seg => {
        confirmedSegments.push(seg);
      });
      try {
        const { results, turn } = await aiProcessText({
          history: conversationHistory,
          texts: batch.map(s => s.text),
        });
        // 成功后追加对话轮次(原样存档,append后永不变,保证请求前缀稳定);
        // 超轮次从头截断(截断处前缀断裂一次全miss,之后恢复)
        conversationHistory.push({ role: "user", content: turn.user });
        conversationHistory.push({ role: "assistant", content: turn.assistant });
        while (conversationHistory.length > HISTORY_MAX_ROUNDS * 2) {
          conversationHistory.shift();
        }
        batch.forEach((seg, i) => {
          seg.opti_text = results[i].optimized;
          seg.en_text = results[i].translated;
          broadcast({ update: seg });
        });
      } catch (error) {
        // 批量失败(已重试耗尽):拆单逐条兜底,不带对话历史(异常路径不追加轮次)
        console.warn("批量处理失败,拆单兜底:", error instanceof Error ? error.message : error);
        for (const seg of batch) {
          try {
            const { results } = await aiProcessText({ history: [], texts: [seg.text] });
            seg.opti_text = results[0].optimized;
            seg.en_text = results[0].translated;
          } catch (e) {
            console.warn("单条兜底也失败,回退原文:", e instanceof Error ? e.message : e);
            seg.opti_text = seg.text;
            seg.en_text = seg.text;
          }
          broadcast({ update: seg });
        }
      }
    }
  } finally {
    draining = false;
  }
}

const filterWords = [
  "明镜与点点",
  "订阅明镜",
  "优优独播剧场",
  "字幕志愿者",
  "不吝点赞",
  "提供的字幕",
  "感谢观看",
  "感谢您的观看",
  "字幕由",
  "李宗盛",
  "词曲",
  "祝福你生日快",
  "祝你生日快",
  "祝我生日快乐",
  "我们下期再见",
  "本次演唱会",
  "歌词翻译成英文字幕",
  "以上言论不代表本台立场",
  "伦桑原创",
  "D, E, F",
  "唱 郑英文",
  "主持人李慧琼",
  "云上工作室",
  "吾皇万岁万岁万万岁",
  "法国族族语",
  "阿 阿 阿 阿 阿 阿",
  "【歌词】",
  "♫♫",
  "MING PAO",
  "现金激励",
  "业务联系",
  "杨建明",
  "订阅我们的频道",
];

export function isValidSegment(s: Segment): boolean {
  return (
    s.text.length > 0 && !filterWords.some((word) => s.text.includes(word))
  );
}

export function clearAllSegments(): void {
  currentSegment = new Segment();
  confirmedSegments = [];
  conversationHistory = [];
  nextSegmentId = 0;
}
