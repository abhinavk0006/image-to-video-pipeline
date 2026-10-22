"""Deterministic school-circuit references and ideal DC schematic animation.

One explicit node/component model drives SVG, PNG and meter readings. This is
an ideal lumped DC model, not a photorealistic apparatus renderer or SPICE.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import math
from pathlib import Path


KINDS = {'wire', 'resistor', 'battery', 'switch', 'ammeter', 'voltmeter'}


def validate(circuit):
    nodes = circuit.get('nodes', {})
    if not nodes or circuit.get('ground') not in nodes:
        raise ValueError('Circuit requires positioned nodes and a ground node')
    for name, point in nodes.items():
        if len(point) != 2 or any(not math.isfinite(float(v)) or v < 0 for v in point):
            raise ValueError(f'Invalid node position: {name}')
    identifiers = set()
    for component in circuit.get('components', []):
        if component.get('kind') not in KINDS or not component.get('id') or component['id'] in identifiers:
            raise ValueError('Every component needs a unique ID and supported kind')
        identifiers.add(component['id'])
        if component.get('n1') not in nodes or component.get('n2') not in nodes or component['n1'] == component['n2']:
            raise ValueError('Component terminals must reference two distinct nodes')
        a, b = nodes[component['n1']], nodes[component['n2']]
        if math.hypot(a[0]-b[0], a[1]-b[1]) < 60:
            raise ValueError('Components need at least 60 layout pixels between terminals')
        if component['kind'] in {'resistor', 'battery'}:
            value = float(component.get('value', 0))
            if not math.isfinite(value) or (component['kind']=='resistor' and value <= 0):
                raise ValueError('Resistors need positive resistance; sources need finite voltage')
        if component['kind'] == 'switch' and not isinstance(component.get('closed', False), bool):
            raise ValueError('Switch closed state must be boolean')
    if not identifiers:
        raise ValueError('Circuit has no components')
    return circuit


def solve_dc(circuit):
    """Modified nodal analysis with ideal meters and ideal voltage sources."""
    import numpy as np
    validate(circuit)
    parent = {node: node for node in circuit['nodes']}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for c in circuit['components']:
        if c['kind']=='wire' or c['kind']=='switch' and c.get('closed', False):
            parent[find(c['n1'])] = find(c['n2'])
    ground = find(circuit['ground'])
    names = sorted({find(node) for node in parent} - {ground})
    indices = {node: i for i, node in enumerate(names)}
    sources = [c for c in circuit['components'] if c['kind'] in {'battery', 'ammeter'}]
    size = len(names)+len(sources)
    matrix, rhs = np.zeros((size, size)), np.zeros(size)
    def idx(node):
        return indices.get(find(node))
    for c in circuit['components']:
        a, b = idx(c['n1']), idx(c['n2'])
        if c['kind']=='resistor':
            conductance = 1/float(c['value'])
            for i in (a, b):
                if i is not None:
                    matrix[i, i] += conductance
            if a is not None and b is not None:
                matrix[a, b] -= conductance
                matrix[b, a] -= conductance
    for number, c in enumerate(sources, start=len(names)):
        if find(c['n1']) == find(c['n2']):
            raise ValueError('Ideal voltage source or ammeter is shorted; revise topology')
        for node, sign in ((c['n1'], 1), (c['n2'], -1)):
            index = idx(node)
            if index is not None:
                matrix[index, number] += sign
                matrix[number, index] += sign
        rhs[number] = float(c.get('value', 0)) if c['kind']=='battery' else 0
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError('Circuit has floating nodes, inconsistent ideal sources or an indeterminate current') from error
    if not np.isfinite(solution).all():
        raise ValueError('Nonfinite circuit solution')
    voltages = {node: 0.0 if find(node)==ground else float(solution[idx(node)]) for node in parent}
    readings = {}
    for c in circuit['components']:
        if c['kind']=='voltmeter':
            readings[c['id']] = {'value': voltages[c['n1']]-voltages[c['n2']], 'unit': 'V'}
        elif c['kind']=='ammeter':
            readings[c['id']] = {'value': float(solution[len(names)+sources.index(c)]), 'unit': 'A'}
    return {'node_voltages': voltages, 'readings': readings,
            'assumptions': 'Ideal DC: zero-resistance wires/closed switches/ammeters, infinite-resistance voltmeters, linear resistors.'}


def draw(circuit, *, readings=None):
    from PIL import Image, ImageDraw, ImageFont
    validate(circuit)
    width, height = map(int, circuit.get('size', [1000, 700]))
    if width < 200 or height < 200 or width > 4096 or height > 4096:
        raise ValueError('Circuit canvas must be between 200 and 4096 pixels per side')
    if width % 2 or height % 2:
        raise ValueError('Circuit video dimensions must be even for yuv420p output')
    if any(not (40 <= p[0] <= width-40 and 80 <= p[1] <= height-80) for p in circuit['nodes'].values()):
        raise ValueError('Circuit nodes must leave label margins inside the canvas')
    image = Image.new('RGB', (width, height), 'white')
    canvas = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
        unicode_font = True
    except OSError:
        font = ImageFont.load_default()
        unicode_font = False
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>']
    color = '#111111'
    def line(points):
        canvas.line(points, fill=color, width=3)
        svg.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="3"/>')
    def text(point, value):
        raster_text = value if unicode_font else value.replace('Ω', 'ohm').replace('−', '-').replace('•', '|')
        canvas.text(point, raster_text, font=font, fill=color, anchor='mm')
        svg.append(f'<text x="{point[0]}" y="{point[1]}" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="20" fill="{color}">{html.escape(value)}</text>')
    def circle(center, radius, fill='white'):
        x,y = center
        canvas.ellipse((x-radius,y-radius,x+radius,y+radius), fill=fill, outline=color, width=3)
        svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="3"/>')
    text((width/2, 32), circuit.get('title', 'Circuit diagram'))
    degrees = {node: 0 for node in circuit['nodes']}
    for c in circuit['components']:
        color = '#1754b8' if c.get('highlight', False) else '#111111'
        a,b = tuple(circuit['nodes'][c['n1']]), tuple(circuit['nodes'][c['n2']])
        degrees[c['n1']] += 1
        degrees[c['n2']] += 1
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        ux,uy = (b[0]-a[0])/length, (b[1]-a[1])/length
        cx,cy = (a[0]+b[0])/2, (a[1]+b[1])/2
        def p(x,y=0):
            return (cx+ux*x-uy*y, cy+uy*x+ux*y)
        kind = c['kind']
        if kind=='wire':
            line([a,b])
        else:
            line([a,p(-25)])
            line([p(25),b])
            if kind=='resistor':
                line([p(-25,-10),p(25,-10),p(25,10),p(-25,10),p(-25,-10)])
            elif kind=='battery':
                line([p(-25),p(-6)])
                line([p(6),p(25)])
                line([p(-6,-22),p(-6,22)])
                line([p(6,-12),p(6,12)])
                text(p(-20,-35), '+')
                text(p(20,-35), '−')
            elif kind in {'ammeter','voltmeter'}:
                circle((cx,cy),25)
                text((cx,cy), 'A' if kind=='ammeter' else 'V')
            elif kind=='switch':
                line([p(-25),p(25,0 if c.get('closed',False) else -22)])
                circle(p(-25),3)
                circle(p(25),3)
            label = c.get('label',c['id'])
            if kind=='resistor':
                label += f"  {float(c['value']):g} Ω"
            if kind=='battery':
                label += f"  {float(c['value']):g} V"
            text((cx+float(c.get('label_dx',0)), cy+float(c.get('label_dy',-52))), label)
            if readings and c['id'] in readings:
                measurement = readings[c['id']]
                text((cx,cy+48), f"{measurement['value']:.3g} {measurement['unit']}")
    color='#111111'
    for node, count in degrees.items():
        if count>=3:
            circle(circuit['nodes'][node],4,fill=color)
    text((width/2,height-28), 'Schematic • ideal DC model • simulated readings')
    svg.append('</svg>')
    return image, ''.join(svg)


def render_reference(circuit, output_stem):
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    image, svg = draw(circuit)
    image.save(stem.with_suffix('.png'))
    stem.with_suffix('.svg').write_text(svg,encoding='utf-8')
    return stem.with_suffix('.png')


def render_animation(circuit, timeline, output, *, seconds=6, fps=16):
    import imageio_ffmpeg
    validate(circuit)
    if not math.isfinite(seconds) or seconds<=0 or fps<=0:
        raise ValueError('Animation duration and FPS must be positive')
    events = sorted(timeline,key=lambda e:float(e['at_seconds']))
    if any(not math.isfinite(float(e['at_seconds'])) or not 0<=float(e['at_seconds'])<seconds for e in events):
        raise ValueError('Timeline event outside animation duration')
    state, position = copy.deepcopy(circuit),0
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    width,height=map(int,state.get('size',[1000,700]))
    writer=imageio_ffmpeg.write_frames(str(output),(width,height),fps=fps,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','18'])
    writer.send(None)
    cached=None
    try:
        for index in range(math.ceil(seconds*fps)):
            changed = cached is None
            while position<len(events) and float(events[position]['at_seconds'])<=index/fps:
                for identifier, updates in events[position]['changes'].items():
                    component=next((c for c in state['components'] if c['id']==identifier),None)
                    if component is None or set(updates)-{'value','closed','highlight'}:
                        raise ValueError('Timeline may only change known component values, switch state or highlighting')
                    component.update(updates)
                position+=1
                changed=True
            if changed:
                solved=solve_dc(state)
                frame,_=draw(state,readings=solved['readings'])
                cached=frame.tobytes()
            writer.send(cached)
    finally:
        writer.close()
    output.with_suffix('.rendering.json').write_text(json.dumps({'render_mode':'circuit_diagram',
        'circuit':circuit,'timeline':timeline,'seconds':seconds,'fps':fps,
        'model':'ideal_dc','photorealistic':False},indent=2),encoding='utf-8')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--circuit',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=6)
    parser.add_argument('--animate',action='store_true')
    args=parser.parse_args()
    payload=json.loads(Path(args.circuit).read_text())
    if args.animate:
        render_animation(payload.get('circuit',payload),payload.get('timeline',[]),args.output,seconds=args.seconds)
    else:
        render_reference(payload.get('circuit',payload),args.output)


if __name__=='__main__':
    main()
