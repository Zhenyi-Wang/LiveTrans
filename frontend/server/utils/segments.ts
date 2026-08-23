import { aiProcessText, aiPreviewInput } from "./ai";
import { broadcast } from "./ws";

// 累积式 context 窗口: 从 min 条起只追加不滑动,请求前缀逐字节稳定以利缓存命中;
// 到达 max 后丢弃到最近 min 条重新累积,循环
const CONTEXT_SEGMENTS_MIN = 10;
const CONTEXT_SEGMENTS_MAX = 30;

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

  async processText(contextSegments: Segment[]): Promise<void> {
    // console.log('---process:', contextSegments, ' >>> ',this.text)
    const result = await aiProcessText(contextSegments, this.text);
    this.opti_text = result.optimized;
    this.en_text = result.translated;
  }

  async previewInput(contextSegments: Segment[]): Promise<void> {
    // console.log('---preview:', contextSegments, ' >>> ',this.text)
    const result = await aiPreviewInput(contextSegments, this.text);
    this.en_text = result.translated;
  }
  

  
  toJSONString() {
    return JSON.stringify(this);
  }
}

let currentSegment: Segment = new Segment();
let confirmedSegments: Segment[] = [];
let contextBase = 0; // 当前累积窗口在 confirmedSegments 中的起点

export function saveCurrentSegment(s: Segment): Segment[] {
  currentSegment = s;
  return confirmedSegments.slice(contextBase);
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
  // 窗口超过 max 后丢弃到最近 min 条(周期性重置,重置轮缓存全 miss)
  if (id - contextBase > CONTEXT_SEGMENTS_MAX) {
    contextBase = id - CONTEXT_SEGMENTS_MIN;
  }
  let contextSegments = confirmedSegments.slice(contextBase);
  confirmedSegments.push(seg);
  return contextSegments;
}

// ===== confirmed 串行批处理队列 =====
// 串行消费保证下一批 context 中历史条目均为翻译完成态(消除竞态,稳定缓存前缀);
// 队列积压时合并为批量调用,每批最多 confirmedBatchMax 条,超出部分循环补齐
const runtimeConfig = useRuntimeConfig();
const CONFIRMED_BATCH_MAX = runtimeConfig.confirmedBatchMax || 2;
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
      // 窗口重置检查 + 取context(此刻全部为完成态)
      if (confirmedSegments.length - contextBase > CONTEXT_SEGMENTS_MAX) {
        contextBase = confirmedSegments.length - CONTEXT_SEGMENTS_MIN;
      }
      const contextSegs = confirmedSegments.slice(contextBase);
      batch.forEach(seg => {
        confirmedSegments.push(seg);
      });
      try {
        const results = await aiProcessText(contextSegs, batch.map(s => s.text));
        batch.forEach((seg, i) => {
          seg.opti_text = results[i].optimized;
          seg.en_text = results[i].translated;
          broadcast({ update: seg });
        });
      } catch (error) {
        // 批量失败(已重试耗尽):拆单逐条兜底,不带context(此时批内条目均为未完成态)
        console.warn("批量处理失败,拆单兜底:", error instanceof Error ? error.message : error);
        for (const seg of batch) {
          try {
            const [r] = await aiProcessText([], [seg.text]);
            seg.opti_text = r.optimized;
            seg.en_text = r.translated;
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
  contextBase = 0;
  nextSegmentId = 0;
}
