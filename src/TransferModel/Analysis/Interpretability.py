import matplotlib as mpl
import numpy as np
from IPython.display import HTML, display
from alibi.explainers import IntegratedGradients

from TransferModel.DataUtils import DataProcessor
from TransferModel.Models.TargetModel import TargetModel


def getIntegratedGradients(model: TargetModel, test_df, method="gausslegendre", n_steps=50, internal_batch_size=32):
    X = model.split_and_pad(test_df["X"])

    preds = model.model_.predict(X, batch_size=model.inference_batch_size_)
    predClasses = np.argmax(preds, axis=1)

    ig = IntegratedGradients(model.model_,
                             layer=model.model_.layers[2],  # Start at embedding layer.
                             target_fn=None,
                             method=method,
                             n_steps=n_steps,
                             internal_batch_size=internal_batch_size)

    explanation = ig.explain(X, target=predClasses)

    return explanation, predClasses, preds


def visualizeExplanation(model, predClasses, preds, explanation, test_df, inputVocab, targetVocab, filename):
    def colorize(attrs, cmap='PiYG'):
        """
        Compute hex colors based on the attributions for a single instance.
        Uses a diverging colorscale by default and normalizes and scales
        the colormap so that colors are consistent with the attributions.
        """
        cmap_bound = np.abs(attrs).max()
        norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
        cmap = mpl.cm.get_cmap(cmap)

        # now compute hex values of colors
        colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
        return colors

    X = model.split_and_pad(test_df["X"])
    forwardsSeq = DataProcessor.unfeaturize_seqs_input(X[0], inputVocab)
    backwardsSeq = DataProcessor.unfeaturize_seqs_input(X[1], inputVocab)

    attrs = explanation.attributions[0].sum(axis=2)

    htmlData = ["<table width: 100%>"]
    rows = [
        "<tr><th>Genbank Accession</th>"
        "<th>True Label</th>"
        "<th>Predicted Label (with probability)</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]

    for i, data in enumerate(zip(forwardsSeq, backwardsSeq, attrs, predClasses, preds, test_df.index)):
        forwardSentence = data[0]
        backwardSentence = data[1]
        attribution = data[2]
        predClass = data[3]
        prob = data[4]
        rowIdx = data[5]
        trueClass = test_df._get_value(rowIdx, 'y')
        genbankID = test_df._get_value(rowIdx, 'Accession')

        print("Row:", rowIdx, "Prediction:", predClass, "attribution:", attribution)
        curColors = colorize(attribution)

        rows.append(
            "".join([
                "<tr>",
                format_classname(genbankID),
                format_classname(DataProcessor.unfeaturize_host(trueClass, targetVocab)),
                format_classname(
                    "{0} ({1:.2f})".format(
                        DataProcessor.unfeaturize_host(predClass, targetVocab), prob[predClass]
                    )
                ),
                format_classname("{0:.2f}".format(sum(attribution))),
                format_word_importances(
                    forwardSentence, curColors
                ),
                "<tr>"
            ])
        )

    htmlData.append("</div>")
    htmlData.append("".join(rows))
    htmlData.append("</table>")
    html = HTML("".join(htmlData))

    with open(filename, 'w') as f:
        f.write(html.data)

    # display(HTML("".join(list(map(hlstr, backwardSentence, curColors)))))


def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_tooltip(item, text):
    return '<div class="tooltip">{item}\
        <span class="tooltiptext">{text}</span>\
        </div>'.format(
        item=item, text=text
    )


def format_word_importances(words, colors):
    if colors is None or len(colors) == 0:
        return "<td></td>"
    assert len(words) <= len(colors)
    tags = ["<td>"]
    for word, color in zip(words, colors):
        word = format_special_tokens(word)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)
