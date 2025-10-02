import { aiProcessText } from "./ai";

const CONTEXT_SEGMENTS_LIMIT = 10;

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
    let context = contextSegments.map((s) => s.opti_text || s.text).join(" ");
    // console.log('---process:', context, ' >>> ',this.text)
    const result = await aiProcessText(context, this.text);
    this.opti_text = result.optimized;
    this.en_text = result.translated;
  }

  
  toJSONString() {
    return JSON.stringify(this);
  }
}

let currentSegment: Segment = new Segment();
let confirmedSegments: Segment[] = [];

export function saveCurrentSegment(s: Segment): Segment[] {
  currentSegment = s;
  return confirmedSegments.slice(
    Math.max(confirmedSegments.length - CONTEXT_SEGMENTS_LIMIT, 0)
  );
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
  let contextSegments = confirmedSegments.slice(
    Math.max(id - CONTEXT_SEGMENTS_LIMIT, 0)
  );
  confirmedSegments.push(seg);
  return contextSegments;
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
}
