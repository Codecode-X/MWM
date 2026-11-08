"""Export the Chinese publication report, including figures and rendered equations.

This script only reads Markdown and existing figures. It never loads a model.
"""
import argparse
import base64
from datetime import datetime, timezone
from html import escape, unescape
from io import BytesIO
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import subprocess
import hashlib


ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def formula(latex, inline=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    value=re.sub(r'\\(mathcal|mathbb)\s+([A-Za-z])',r'\\\1{\2}',latex.strip().replace('\n',' '))
    stream=BytesIO()
    math_to_image('$'+value+'$',stream,format='svg',
                  prop=FontProperties(size=12 if inline else 14,math_fontfamily='stix'))
    encoded=base64.b64encode(stream.getvalue()).decode()
    css_class='math-inline' if inline else 'math-display'
    image=f'<img class="{css_class}" src="data:image/svg+xml;base64,{encoded}" alt="{escape(latex,quote=True)}" />'
    return image if inline else '<div class="equation">'+image+'</div>'


def html_body(source):
    from markdown_it import MarkdownIt
    from PIL import Image
    text=source.read_text()
    title=text.splitlines()[0].lstrip('# ').strip()
    text=re.sub(r'<!--.*?-->','',text,flags=re.S)
    text=re.sub(r'\\\[(.*?)\\\]',lambda m:formula(m.group(1)),text,flags=re.S)
    text=re.sub(r'\\\((.*?)\\\)',lambda m:formula(m.group(1),True),text,flags=re.S)
    body=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
    body=re.sub(r'<h1>.*?</h1>','',body,count=1,flags=re.S)
    headings=[]

    def heading(match):
        level,content=match.groups();ident=f'section-{len(headings)+1}'
        headings.append((int(level),re.sub(r'<[^>]+>','',unescape(content)),ident))
        return f'<h{level} id="{ident}">{content}</h{level}>'

    body=re.sub(r'<h([23])>(.*?)</h\1>',heading,body,flags=re.S)
    figures=[]

    def figure(match):
        image,caption=match.groups()
        src=unescape(re.search(r'src="([^"]+)"',image).group(1))
        path=(source.parent/src).resolve()
        with Image.open(path) as picture:
            width,height=picture.size
        number=int(re.search(r'<strong>图 (\d+)',caption).group(1))
        orientation='portrait' if height/width>.88 else 'wide'
        figures.append({'number':number,'path':str(path.relative_to(ROOT)),
                        'sha256':sha(path),'orientation':orientation})
        image=image.replace('src="'+escape(src,quote=True)+'"','src="'+path.as_uri()+'"')
        return (f'<figure class="report-figure {orientation}" id="figure-{number}">'+image+
                f'<figcaption><p>{caption}</figcaption><span id="figure-end-{number}"></span></figure>')

    body=re.sub(r'<p>(<img[^>]+>)</p>\s*<p>(<strong>图 \d+.*?</p>)',figure,body,flags=re.S)
    assert len(figures)==60 and sorted(x['number'] for x in figures)==list(range(1,61))
    table_count=0

    def table(match):
        nonlocal table_count
        title,content,caption=match.groups();table_count+=1
        columns=len(re.findall(r'<th(?:\s|>)',content))
        cls='compact' if columns>=8 else 'standard'
        return f'<div class="table-block {cls}">{title or ""}{content}{caption or ""}</div>'

    body=re.sub(r'(<p>表 \d+：.*?</p>\s*)?(<table>.*?</table>)\s*(<p><strong>(?:表|设置表|执行表|进度表).*?</p>)?',table,body,flags=re.S)
    assert table_count==34,table_count
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

    def link(match):
        href=unescape(match.group(1))
        if href.startswith(('#','https:','http:','mailto:')):
            return match.group(0)
        target=(source.parent/href).resolve()
        try:
            rel=target.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)
        kind='tree' if target.is_dir() else 'blob'
        return 'href="'+escape(f'https://github.com/Codecode-X/MWM/{kind}/{revision}/{rel}',quote=True)+'"'

    body=re.sub(r'href="([^"]+)"',link,body)
    toc='<nav class="toc"><h2>目录</h2><ol>'
    for level,text,ident in headings:
        toc+=f'<li class="level-{level}"><a href="#{ident}">{escape(text)}</a></li>'
    toc+='</ol></nav>'
    cover=(f'<section class="cover"><div class="eyebrow">ADNI · MIMIC · CausalMed</div>'
           f'<h1>{escape(title)}</h1><p>实验设置、定量比较与动力学分析</p>'
           '<p class="cover-summary">89 项已完成实验与分析<br>60 张研究图 · 34 张结果与设置表</p>'
           f'<p class="cover-source">文档版本：{revision[:10]}<br>导出日期：{datetime.now(timezone.utc).date()}</p></section>')
    return title,cover+toc+body,figures,table_count,revision


CSS=r'''
@page { size: A4 landscape; margin: 15mm 15mm 16mm;
  @top-left { content: "纵向临床模型实验报告"; font: 8pt "Noto Sans CJK SC"; color: #718096; }
  @top-right { content: "ADNI · MIMIC · CausalMed"; font: 8pt "Noto Sans CJK SC"; color: #718096; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt "Noto Sans CJK SC"; color: #718096; }
}
@page portrait { size: A4 portrait; margin: 15mm; }
@page cover { @top-left {content: none} @top-right {content: none} }
html { font-family: "Noto Sans CJK SC", "DejaVu Sans", sans-serif; color: #202d3c; }
body { font-size: 10.5pt; line-height: 1.65; }
h1,h2,h3 { color: #15324c; line-height: 1.4; break-after: avoid; }
h1 { font-size: 25pt; } h2 { font-size: 17pt; margin: 9mm 0 4mm; }
h3 { font-size: 13.5pt; margin: 6mm 0 3mm; }
p { margin: 0 0 3mm; orphans: 3; widows: 3; }
a { color: #246684; text-decoration: none; overflow-wrap: anywhere; }
strong { color: #15324c; }
code,pre { font-family: "Noto Sans Mono CJK SC", monospace; font-size: 9pt; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f7fa; padding: 3mm; }
ul,ol { padding-left: 6mm; }
.equation { text-align: center; margin: 5mm 0; break-inside: avoid; }
.equation img { max-width: 100%; height: auto; }
img.math-inline { height: 1.15em; width: auto; vertical-align: -.15em; }
.table-block { break-inside: avoid; margin: 4mm 0 5mm; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; line-height: 1.5; margin-bottom: 3mm; }
.compact table { font-size: 8pt; }
th { background: #e8eff5; color: #15324c; font-weight: 700; text-align: left; }
th,td { border: .3pt solid #c4d0dc; padding: 1.4mm 1.7mm; vertical-align: top; overflow-wrap: anywhere; }
thead { display: table-header-group; } tr { break-inside: avoid; }
tbody tr:nth-child(even) { background: #f8fafc; }
.table-block>p:last-child { font-size: 9pt; color: #435164; line-height: 1.6; }
.report-figure { page: auto; break-before: page; break-after: page; break-inside: avoid; margin: 0; padding: 2mm 0; }
.report-figure>img { display: block; width: auto; height: auto; max-width: 100%; max-height: 132mm; margin: 0 auto 4mm; }
.report-figure.portrait { page: portrait; }
.report-figure.portrait>img { max-height: 208mm; }
figcaption { font-size: 10pt; line-height: 1.6; padding-top: 2mm; border-top: .5pt solid #d8e0e8; }
figcaption p { margin: 0; }
.cover { page: cover; break-after: page; padding: 20mm 10mm 0; }
.cover h1 { max-width: 225mm; margin: 7mm 0 10mm; font-size: 29pt; }
.eyebrow { font-size: 16pt; letter-spacing: 1pt; color: #246684; }
.cover-summary { margin-top: 13mm; color: #435164; line-height: 1.9; }
.cover-source { margin-top: 13mm; color: #718096; font-size: 9pt; }
.toc { break-after: page; } .toc h2 { margin-top: 0; }
.toc ol { columns: 2; column-gap: 13mm; list-style: none; padding-left: 0; }
.toc li { break-inside: avoid; font-size: 9.5pt; line-height: 1.65; margin-bottom: 1.2mm; }
.toc .level-2 { font-weight: bold; margin-top: 3mm; } .toc .level-3 { padding-left: 3mm; }
.toc a::after { content: leader('.') target-counter(attr(href), page); }
'''


def export(source,output):
    from weasyprint import HTML
    import pymupdf as fitz
    source=source.resolve();output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    title,body,figures,table_count,revision=html_body(source)
    html=(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
          f'<title>{escape(title)}</title><meta name="author" content="LatentODE / MWM">'
          f'<style>{CSS}</style></head><body>{body}</body></html>')
    warnings=[]

    class Capture(logging.Handler):
        def emit(self,record):
            if record.levelno>=logging.WARNING:
                warnings.append(record.getMessage())

    handler=Capture();logger=logging.getLogger('weasyprint');logger.addHandler(handler)
    try:
        rendered=HTML(string=html,base_url=source.parent.as_uri()+'/').render()
        pages={name:index+1 for index,page in enumerate(rendered.pages) for name in page.anchors}
        for figure in figures:
            ident=f'figure-{figure["number"]}'
            assert pages[ident]==pages[f'figure-end-{figure["number"]}'],f'Split caption: {ident}'
            figure['pdf_page']=pages[ident]
        rendered.write_pdf(output)
    finally:
        logger.removeHandler(handler)
    pdf=fitz.open(output);texts=[page.get_text() for page in pdf];all_text='\n'.join(texts)
    for required in ['模型定义与共同实验标准','MIMIC','浅灰色空心圆','0.991104','1.795789']:
        assert required in re.sub(r'\s+','',all_text),required
    assert '\ufffd' not in all_text,'Replacement character found'
    assert len(pdf.get_toc())>=35,'Missing PDF bookmarks'
    empty=[];outside=[]
    for i,page in enumerate(pdf):
        body=[]
        for block in page.get_text('blocks'):
            bounds=fitz.Rect(block[:4]);value=block[4].strip()
            if bounds.x0<-.5 or bounds.y0<-.5 or bounds.x1>page.rect.width+.5 or bounds.y1>page.rect.height+.5:
                outside.append(i+1)
            if bounds.y0>32 and bounds.y1<page.rect.height-30:
                body.append(value)
        if len(''.join(body))<10 and not page.get_image_info():
            empty.append(i+1)
    assert not empty,empty
    assert not outside,outside
    report={'status':'complete','exported_at_utc':datetime.now(timezone.utc).isoformat(),
            'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'source_commit':revision,
            'pdf':str(output.relative_to(ROOT)),'pdf_sha256':sha(output),'pdf_bytes':output.stat().st_size,
            'pages':len(pdf),'figures':len(figures),'tables':table_count,'bookmarks':len(pdf.get_toc()),
            'all_figure_captions_on_same_page':True,'chinese_text_extraction_verified':True,
            'blank_pages':empty,'text_blocks_outside_pages':outside,
            'model_execution_performed':False,'warnings':warnings,'figure_pages':figures,
            'renderer_versions':{k:importlib.metadata.version(k) for k in ['weasyprint','PyMuPDF','markdown-it-py','matplotlib','Pillow']}}
    output.with_suffix('.pdf.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k!='figure_pages'},indent=2,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.md')
    parser.add_argument('--output',type=Path,default=ROOT/'reproduction/NO_WEAK_PRIOR_PUBLICATION_RERUN.zh-CN.pdf')
    args=parser.parse_args();export(args.source,args.output)
