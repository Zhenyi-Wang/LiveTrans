import * as OpenCC from 'opencc-js';

const converter = OpenCC.Converter({ from: 'hk', to: 'cn' });

export function cnT2S(text: string): string {
  const result = converter(text);
  return result;
}
