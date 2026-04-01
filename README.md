# Hyperbase Alpha-Beta Parser

## A semantic hypergraph parser for natural language

The Alpha-Beta parser is a [Hyperbase](https://hyperquest.ai/hyperbase) plugin that converts natural language text into *Semantic Hypergraphs (SH)*. It works in two stages:

- **Alpha stage**: A multilingual neural token classifier (based on DistilBERT) assigns one of 39 semantic atom types to each token in a sentence -- for example, concepts, predicates, modifiers, builders, triggers and conjunctions.
- **Beta stage**: A rule-based engine combines classified atoms into ordered, recursive hyperedges using syntactic and semantic composition rules, producing structured representations that can be manipulated with Hyperbase.

## Supported languages

The parser supports any language with a [spaCy](https://spacy.io) model available, including English, French, German, Italian, Portuguese and Spanish.

While the parser is theoretically language-agnostic and could in principle support languages such as Mandarin, which differ substantially in morphological and syntactic structure, the authors' linguistic competence is limited to Germanic and Romance languages. We welcome the help of native speakers or domain experts in validating/improving support for other language families.

## Installation and manual

Installation instructions, the manual and more information can be found here: <https://hyperquest.ai/hyperbase>

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
