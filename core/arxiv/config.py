categories = [
    'cs.CL', 'cs.NE', 'physics.comp-ph', 
    'q-bio.BM', 'eess.AS', 'cs.MM', 'math.IT', 
    'q-bio.QM', 'I.2.10; I.4.8; I.2.6; I.2.7; I.5.4; I.5.1', 
    'physics.chem-ph', 'cs.SD', 'cs.CV', 'cs.AR', 
    'cond-mat.soft', 'cond-mat.mtrl-sci', 'cs.RO', 
    'cs.MA', 'I.2.1', 'cs.IT', 'cs.HC', 'eess.IV', 
    'cs.IR', 'cs.AI', 'cs.CY', 'I.4.9', 'cs.LG', 'cs.NI', 
    'cond-mat.stat-mech', 'cs.DC'
]

CATEGORIES_QUERY_STRING = f"({' OR '.join([f'cat:{cat}' for cat in categories])})"