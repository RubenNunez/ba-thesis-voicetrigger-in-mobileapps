import pandas as pd
from io import StringIO

# CSV data (truncated for brevity)
csv_data = """
Person;Word;Result;Comment
Person 1;HEY_FOOBY;predicted/not predicted
Person 1;HEY_FOOBY;predicted
Person 1;HEY_FOOBY;not predicted
Person 1;HEY_FOOBY;predicted
Person 1;HEY_FOOBY;not predicted
Person 1;HEY_FOOBY;predicted
Person 1;HEY_FOOBY;not predicted
Person 1;HEY_FOOBY;predicted
Person 1;HEY_FOOBY;not predicted
Person 1;HEY_FOOBY;not predicted
Person 1;HEY_FOOBY;not predicted
..;..;..;...
Person 1;other;not predicted
Person 1;other;not predicted
Person 1;other;not predicted
Person 1;other;predicted
Person 1;other;predicted
Person 1;other;predicted
Person 1;other;not predicted
Person 1;other;not predicted
Person 1;other;not predicted
Person 1;other;not predicted
..;..;..;Wenn deutlich gesprochen wird funktioniert es gut.
..;..;..;...
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;not predicted
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;predicted
Person 2;HEY_FOOBY;not predicted
..;..;..;...
Person 2;other;not predicted
Person 2;other;not predicted
Person 2;other;predicted;(SMA-3)
Person 2;other;not predicted
Person 2;other;not predicted
Person 2;other;not predicted
Person 2;other;not predicted
Person 2;other;predicted;hat Hey Boddy gesagt
Person 2;other;not predicted
Person 2;other;not predicted
..;..;..;Relativ Stabil könnte aber besser sein
..;..;..;...
Person 3;HEY_FOOBY;predicted
Person 3;HEY_FOOBY;predicted
Person 3;HEY_FOOBY;predicted
Person 3;HEY_FOOBY;predicted
Person 3;HEY_FOOBY;predicted;(SMA 3)
Person 3;HEY_FOOBY;predicted
Person 3;HEY_FOOBY;predicted
Person 3;HEY_FOOBY;not predicted
Person 3;HEY_FOOBY;predicted
Person 3;HEY_FOOBY;predicted;(SMA 3)
..;..;..;...
Person 3;other;predicted
Person 3;other;not predicted
Person 3;other;predicted
Person 3;other;predicted;beim sageb von "alles andere"
Person 3;other;not predicted
Person 3;other;predicted;beim Husten
Person 3;other;predicted;(SMA 3)
Person 3;other;predicted;beim sagen von "Erdbertörtli"
Person 3;other;not predicted
Person 3;other;not predicted
Person 3;other;not predicted
Person 3;other;not predicted
Person 3;other;not predicted
..;..;..;Seine Stimme war im Trainingset enthalten. Sein Fazit: Am Anfang hat es nicht funktioniert, dann aber schon. Funktioniert schon nicht zu 100%. Für Produktivbetrieb nicht geeignet. Wenn in Befehlsform gesprochen wird, werden falsche Wörter erkannt. Ganze Sätze werden aber zuverläsig erkannt.
..;..;..;...
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted;(SMA 3)
Person 4;HEY_FOOBY;predicted
Person 4;HEY_FOOBY;predicted
..;..;..;...
Person 4;other;predicted;(SMA 3)
Person 4;other;not predicted
Person 4;other;not predicted
Person 4;other;not predicted
Person 4;other;not predicted
Person 4;other;predicted;(SMA 3) hat "Hey der Zug ist zuspät" gesagt
Person 4;other;not predicted
Person 4;other;predicted;(SMA 3)
Person 4;other;predicted;"Hey" war enthalten
Person 4;other;not predicted
..;..;..;Sehr zufriedenstellend. Funktioniert gut. Bei ähnlichen Wörtern wird das falsche Wort erkannt. "Daumen hoch"
..;..;..;...
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
Person 5;HEY_FOOBY;predicted
..;..;..;...
Person 5;other;not predicted
Person 5;other;predicted;(SMA 3)
Person 5;other;predicted;beim sagen von "Gemüsesuppe"
Person 5;other;not predicted
Person 5;other;not predicted
Person 5;other;predicted;(SMA 3)
Person 5;other;predicted
Person 5;other;predicted;Satz hat "FOOBY" enthalten
Person 5;other;predicted;(SMA 3)
Person 5;other;predicted;(SMA 3)
..;..;..;Es funktioniert relativ gut aber viele falsche Vorhersagen in der other Klasse.
..;..;..;...
Person 6;HEY_FOOBY;predicted
Person 6;HEY_FOOBY;predicted
Person 6;HEY_FOOBY;not predicted
Person 6;HEY_FOOBY;predicted
Person 6;HEY_FOOBY;not predicted
Person 6;HEY_FOOBY;predicted
Person 6;HEY_FOOBY;predicted
Person 6;HEY_FOOBY;predicted
Person 6;HEY_FOOBY;predicted
Person 6;HEY_FOOBY;predicted
..;..;..;...
Person 6;other;not predicted
Person 6;other;predicted;(SMA 3)
Person 6;other;predicted;beim sagen von "Gemüsesuppe"
Person 6;other;not predicted
Person 6;other;not predicted
Person 6;other;predicted;(SMA 3)
Person 6;other;not predicted
Person 6;other;predicted;Satz hat "FOOBY" enthalten
Person 6;other;predicted;(SMA 3)
Person 6;other;predicted;(SMA 3)
..;..;..;Es erkennt andere Wörter als Triggerwort. Die negativen Beispiele sind nicht gut.
"""

# Read the CSV data into a pandas DataFrame
df = pd.read_csv(StringIO(csv_data), delimiter=";")

# Convert the DataFrame to a LaTeX table
latex_table = df.to_latex(index=False)

print(latex_table)

