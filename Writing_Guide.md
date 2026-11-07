# Scientific Writing Style Guide
## For Astrophysics and Technical Research Manuscripts

### 1. Overall writing philosophy

Write in a clear, concise, and natural academic style suitable for a research paper in astrophysics, particularly the methods sections of journals such as *The Astrophysical Journal* (ApJ), *Monthly Notices of the Royal Astronomical Society* (MNRAS), and *Astronomy & Astrophysics* (A&A).

The writing should be **physically motivated, mathematically precise, straightforward, and fluid**.

The goal is to explain complex scientific ideas without making the language unnecessarily complex. The reader should understand the physical problem, why a particular mathematical treatment is needed, and what the resulting equations mean.

Prioritize clarity over sophistication of language. Do not simplify the science, omit necessary assumptions, or sacrifice mathematical accuracy.

The preferred progression is

**Physical motivation → modeling choice → mathematical formulation → physical interpretation → validation.**

These elements should form a continuous explanation rather than a series of disconnected statements.

### 2. Sentence structure and flow

Use direct, natural scientific prose.

- Write one main idea per sentence.
- Prefer short to medium-length sentences, but vary their length to maintain a natural rhythm.
- Combine closely related ideas when separating them would make the writing choppy.
- Use a new line for each sentence in the LaTeX source.
- Connect sentences logically so that each develops the previous idea.
- Avoid repeating the same point in slightly different words.
- Use active voice when it makes the sentence clearer.
- Prefer simple verbs such as *use, describe, assume, define, calculate, infer, account for, compare,* and *estimate*.

**Avoid excessively short, disconnected sentences.**

Poor

> The source flux is unknown. We introduce a nuisance parameter. We use a flat prior. We integrate over the parameter. The result is Gaussian.

Preferred

> The intrinsic source flux is unknown, so we treat it as a nuisance parameter and marginalize over it analytically. With a flat prior, the Gaussian integral can be evaluated in closed form.

**Avoid overly elaborate academic language.**

Poor

> In order to appropriately characterize the uncertainties arising from the inherent limitations of the adopted gravitational lensing mass distribution, we introduce an additional phenomenological parameterization.

Preferred

> We introduce an additional scatter to account for errors not captured by the lens model.

Use transitions such as *because, since, therefore, however, to account for this, following,* and *this allows us to* when they help the argument. Do not insert a transition into every sentence.

Use colons and semicolons when appropriate, but prefer ordinary sentence structure when it is clearer.

### 3. Structure of a subsection

A technical subsection should generally follow this order.

**A. Physical premise**

Begin with one or two sentences explaining what is observed and what the model must explain.

For example

> Multiple images of the same background source share a common intrinsic brightness, while their observed fluxes differ because of gravitational magnification.

Avoid opening with an equation or a list of parameters before establishing their physical purpose.

**B. Mathematical model**

Introduce the main relation after explaining why it is needed.

Define the symbols that appear in the equation, preferably in the surrounding prose.

Explain any nuisance parameters and how they are treated, including whether they are sampled, fixed, or marginalized analytically.

**C. Uncertainty model**

Introduce uncertainties one at a time.

For each contribution, explain

1. Where the uncertainty comes from physically.
2. The plane or space where it is naturally described.
3. The parameter or covariance representing it.
4. Whether the relevant scale is fixed or inferred.

For lensing applications, use the principle that an uncertainty is simplest in the plane where it originates. For example, image-position measurement errors are naturally defined in the image plane, while residual deflection errors enter the backprojected source position directly.

This is a modeling framework, not a claim that every uncertainty is necessarily isotropic or independent.

**D. Combined model**

After introducing the individual contributions, combine them in one clearly labeled equation.

Avoid repeatedly writing the same total covariance or likelihood.

Explain the important limiting behavior immediately after the equation. Where relevant, discuss ordinary images, highly magnified images, and critical curves, followed by positivity or nonsingularity.

**E. Likelihood construction**

Introduce the likelihood through the full joint probability model.

Write the product over the relevant observations before marginalizing nuisance parameters or simplifying the expression.

Make the conditional independence assumptions clear and retain any shared latent variables or correlations.

Then introduce the marginalization integral, derive or state the simplified likelihood, and explain its normalization.

**F. Closing paragraph**

End with a short statement identifying the new inferred parameters, the fixed observational inputs, and where the method is validated.

Do not repeat the motivation or summarize the entire subsection.

### 4. Introducing equations

Every important equation should have a clear purpose.

**Explain why the equation is needed before presenting it.**

Preferred

> We account for residual errors in the predicted deflections using an isotropic source-plane scatter \(\sigma_{\rm def}\), with covariance

\[
\mathbf C_{i,\rm def}=\sigma_{\rm def}^2\mathbf I.
\]

Avoid introducing equations with vague statements such as

> We can now write the following expression.

Use natural mathematical transitions.

- "We write"
- "We define"
- "The corresponding covariance is"
- "The total variance is"
- "Marginalizing over this parameter yields"
- "The resulting likelihood is"
- "Along each principal direction, this becomes"

Do not use "gives" repeatedly when "yields," "becomes," "reduces to," or "is" would be more precise.

After an equation, explain only what the reader needs to understand its physical meaning or consequences.

Avoid narrating algebra that is already obvious from the equation.

### 5. Mathematical precision

Maintain strict mathematical correctness.

- Define every symbol at first use.
- State whether a quantity is a scalar, vector, or matrix when this is not clear.
- Distinguish variances from standard deviations.
- Specify units when they matter.
- Identify dimensionless parameters.
- Keep image-plane and source-plane quantities distinct.
- Use consistent notation throughout the manuscript.
- State assumptions of independence, isotropy, Gaussianity, or linearity when they are introduced.
- State the conditions under which an approximation is valid.
- Preserve normalization terms that depend on model parameters.
- Do not describe a locally normalized approximation as globally normalized without proof.

When introducing a new parameter, explain what it controls physically rather than merely calling it a nuisance parameter.

For example

> The dimensionless parameter \(\lambda_{\rm sel}\) controls how the selection uncertainty changes with the local image stretching.

Do not introduce additional intermediate symbols unless they substantially simplify the derivation.

Prefer reusing existing quantities over defining symbols that appear only once.

### 6. Explaining approximations and limiting behavior

Approximations should be stated clearly and without unnecessary qualifications.

For example

> We approximate the lens mapping as linear over the positional uncertainty.

When possible, state the condition under which the approximation holds.

If a numerical threshold is important, report it only when supported by a derivation, measurement, or validation test.

Explain limiting behavior in a consistent order.

1. Ordinary or weakly distorted regime.
2. Strongly magnified or critical regime.
3. Positivity, finiteness, or normalization, where relevant.

For example

> Away from critical curves, the smoothed determinant approaches the usual Jacobian area factor. Near critical curves, averaging over positional uncertainty keeps the local approximation finite when the conditional determinant variance is positive.

Do not claim that a formula is exact at a critical curve merely because one part of its derivation is exact.

Distinguish algebraic identities, modeling assumptions, approximations, and numerical findings.

### 7. Physical interpretation

Explain mathematical results in terms of observable effects whenever possible.

For strong lensing, relate equations to image positions, arc stretching, magnification, source reconstruction, and uncertainties.

For example

> A small displacement along a strongly stretched arc can correspond to a much smaller displacement in the source plane than the same displacement across the arc.

This is preferable to introducing a long discussion of singular vectors or quadratic forms when those details have already been established elsewhere.

Use mathematical terminology when necessary, but do not introduce it for its own sake.

Avoid terms such as *Gram matrix, parallel sum, Woodbury identity,* or *regularized structure tensor* in the main text unless the concept is essential to understanding the method.

Detailed derivations and implementation notes belong in appendices or technical documentation.

### 8. Literature and citations

Integrate citations naturally into the argument.

Cite previous work at the point where its specific contribution is discussed.

When adapting an established method, explain briefly what is retained and what differs.

For example

> Following \citet{Raney2021}, we account for residual deflection errors directly in the source plane, but infer an isotropic scatter rather than using a simulation-derived covariance.

Avoid lengthy literature reviews inside the mathematical derivation.

Do not cite papers merely because they discuss the same general subject. Verify that they support the specific claim.

Distinguish established results from new modeling prescriptions.

Do not claim novelty solely because an identical equation has not been found. Novelty claims should reflect a sufficiently thorough literature review and clearly identify the proposed contribution.

### 9. Treatment of uncertainties

Keep the physical origin of each uncertainty clear.

For the current lensing framework, the terminology is as follows.

**Measurement uncertainty, \(\sigma_{\rm meas}\)**

Describes errors in determining an observed image position, including image alignment and manual or automated position measurements.

**Residual deflection scatter, \(\sigma_{\rm def}\)**

Describes errors in predicted deflections that are not represented by the sampled lens-model parameters.

Do not include uncertainty from explicitly sampled member-galaxy deviations a second time.

**Selection scatter, \(\sigma_{\rm sel}\)**

Describes uncertainty in choosing corresponding positions within extended source features.

Distinguish this from measurement uncertainty. A position can be measured accurately while referring to a slightly different physical location in the source.

**Fixed instrumental uncertainty, \(\sigma_{\rm inst}\)**

An observational input used to define the effective measurement variance.

**Additional measurement scatter, \(\sigma_{\rm extra}\)**

An inferred contribution accounting for positional errors beyond the fixed instrumental estimate.

When relevant, write

\[
\sigma_{\rm meas}^2
=
\sigma_{\rm inst}^2+\sigma_{\rm extra}^2.
\]

Use "inferred scalar" for a sampled uncertainty scale and "fixed per image" for observational inputs when those distinctions matter.

Avoid the generic label "systematic uncertainty" when a more specific physical description is available.

### 10. Likelihoods and probability distributions

**Always introduce the full joint likelihood before marginalizing or simplifying it.**

This is a central rule for writing likelihood subsections.

The reader should first see how the individual observations contribute to the probability model, before any nuisance parameters are removed or algebraic simplifications are made.

**A. Begin with the full product**

Write the likelihood using explicit product notation, `\prod`, over the relevant observations, images, source families, or measurement bands.

For example

\[
\mathcal L
=
\prod_f\prod_{i\in f}
p(\boldsymbol d_i\mid\boldsymbol\eta,\boldsymbol\beta_f),
\]

where \(\boldsymbol d_i\) denotes the observed data for image \(i\), \(\boldsymbol\eta\) contains the lens-model parameters, and \(\boldsymbol\beta_f\) is the source position of family \(f\).

This makes the contribution of each observation clear and establishes the assumed conditional independence structure.

**B. Introduce the uncertainties and nuisance parameters**

Explain the probability distributions entering each factor.

Identify which quantities are observed, fixed, sampled, or marginalized.

If multiple observations depend on a shared nuisance parameter, keep that parameter explicit in the joint likelihood.

**C. Write the marginalization integral**

If a nuisance parameter is marginalized, show the integral over its prior.

For a shared parameter \(M_f\), for example,

\[
\mathcal L_f
=
\int dM_f\,p(M_f)
\prod_{i\in f}
p(d_i\mid M_f,\boldsymbol\eta).
\]

The product belongs inside the integral because the observations share the same unknown parameter.

Do not marginalize each observation independently when the nuisance parameter is shared.

**D. Present the simplified likelihood**

After establishing the integral, explain how it is evaluated.

Use a short prose description of the mathematical step, such as

> With a flat prior on the source magnitude, the Gaussian integral can be evaluated analytically, yielding

Then present the closed form.

Keep intermediate algebra out of the main text unless it is necessary to understand the result.

**E. Explain normalization and assumptions**

Briefly explain the meaning of the resulting terms and retain normalization factors that depend on sampled parameters.

If an improper prior is used, distinguish a likelihood defined up to a parameter-independent constant from a fully normalized marginal probability density.

If the likelihood relies on a local approximation, state that clearly.

**F. Preserve dependencies and correlations**

The product form must reflect the actual probability model.

Do not factorize correlated observations as independent.

When observations share correlated errors, latent variables, or selection effects, retain those dependencies in the joint distribution.

Use a multivariate density when that is the correct representation.

**G. Avoid starting with the final chi-square**

Do not begin a likelihood derivation directly with a marginalized chi-square, covariance-weighted residual, or final closed-form expression.

Establish the underlying probability model first, then show how the simplified expression follows.

The purpose of the initial product is to ground the reader, not to introduce unnecessary algebra.

### 11. Numerical validation and verification

Report completed verification as a result, not a promise.

Distinguish among

- Algebraic verification.
- Numerical agreement with a reference calculation.
- Parameter-recovery tests.
- Posterior uncertainty calibration.
- Computational performance.

These establish different properties and should not be treated as interchangeable.

For example

> We verified the approximation against direct numerical integration, including positions near critical curves. The largest differences arise where the local linear approximation is least accurate.

When reporting numerical accuracy, specify the metric, conditions, and result if available.

Do not invent numerical thresholds or claim that an approximation is accurate everywhere based on a limited test.

When discussing computational advantages, use restrained language such as *fast, differentiable,* or *avoids image finding* only when those properties follow from the method or have been verified.

### 12. LaTeX formatting conventions

Use publication-ready LaTeX.

- Put each sentence on a separate source line.
- Keep related sentences in the same paragraph unless there is a meaningful change in topic.
- Use `equation` for single numbered equations.
- Use `aligned` inside `equation` for multiline expressions.
- Label important equations and refer to them consistently with `Equation~\ref{...}`.
- Avoid numbering intermediate expressions that are never referenced.
- Use consistent bold notation for vectors and matrices.
- Keep scalar parameters unbolded.
- Use `\rm` or the manuscript's established roman-subscript convention consistently.
- Prefer compact equations that fit a two-column journal layout.
- Avoid introducing symbols solely to shorten an equation unless doing so improves readability.
- Keep mathematical punctuation consistent with the surrounding sentence.

When editing an existing passage, preserve its equation labels and notation unless a change is necessary.

When asked for a revision, provide the replacement LaTeX directly rather than rewriting unrelated sections.

### 13. Language and expressions to avoid

Avoid unnecessarily formal or inflated expressions, including

- "It is worth noting that"
- "It is important to emphasize that"
- "In order to"
- "For the purpose of"
- "It can readily be shown that"
- "This naturally gives rise to"
- "The aforementioned"
- "A comprehensive treatment of"
- "It is evident that"
- "One may straightforwardly demonstrate"

Prefer direct alternatives.

| Avoid | Prefer |
|---|---|
| In order to account for | To account for |
| It can be shown that | We find that |
| The expression gives | The expression yields |
| We seek to construct | We use |
| It is worth noting that | State the fact directly |
| In the vicinity of | Near |
| With respect to | Relative to |
| A non-negligible contribution | A significant contribution, when supported |
| A parameterization is introduced | We introduce |
| The uncertainty is characterized by | We describe the uncertainty using |

Do not eliminate technical terminology when it is the most precise way to express the idea.

### 14. Avoid unnecessary complexity

Do not introduce additional parameters, thresholds, switches, or functions without a clear physical or statistical reason.

Prefer a dependence derived from quantities already in the model.

Do not add regularization solely to make a numerical problem disappear without explaining how it changes the statistical model.

When two formulations are mathematically equivalent, choose the one that is easier to interpret and implement.

When a simpler model loses an important physical distinction, explain the tradeoff rather than silently removing it.

### 15. Preserve the author's voice

When revising existing text

- Keep the original scientific meaning.
- Preserve the author's terminology unless it is ambiguous or incorrect.
- Maintain a straightforward and restrained tone.
- Improve transitions without making the prose longer than necessary.
- Avoid replacing familiar words with more sophisticated synonyms.
- Do not introduce new claims merely to improve the flow.
- Retain citations unless they are incorrect or no longer support the revised sentence.
- Preserve mathematical notation and equation labels wherever possible.
- Make targeted edits rather than rewriting an entire section when only a few sentences need improvement.

The revised passage should sound like a clearer version of the author's writing, not like a different author.

### 16. Final editing checklist

Before returning a manuscript revision, verify the following.

**Scientific accuracy**

- Are all equations correct?
- Are all symbols defined?
- Are the units and dimensions consistent?
- Are the assumptions stated?
- Are approximations distinguished from exact results?
- Are claims supported by the cited literature?
- Are numerical verification claims actually established?

**Likelihood construction**

- Is the full joint likelihood introduced using explicit product notation?
- Are the relevant observations and conditioning variables defined?
- Does the product reflect the correct conditional independence structure?
- Are shared nuisance parameters kept inside the joint likelihood before marginalization?
- Is the marginalization integral shown before the simplified result?
- Are normalization factors and parameter-dependent terms retained?
- Is the reference measure of the probability density clear?
- Are local approximations distinguished from exact likelihoods?

**Clarity**

- Does the paragraph begin with the physical reason for the model?
- Does each sentence follow naturally from the previous one?
- Is the equation introduced before it appears?
- Is the physical interpretation clear?
- Is any explanation repeated?
- Can a sentence be shortened without losing meaning?

**Consistency**

- Are image-plane and source-plane quantities distinguished?
- Are fixed and inferred parameters identified correctly?
- Is the notation consistent with earlier sections?
- Are equation labels preserved?
- Are the limiting cases described correctly?

**Style**

- Is the prose straightforward and natural?
- Are sentences connected without being overly long?
- Is technical jargon used only when necessary?
- Are citations integrated smoothly?
- Does the passage avoid overstating novelty, accuracy, or importance?

### 17. Example of the target writing style

The following illustrates the preferred progression from physical motivation to a mathematical model.

Multiple images of the same source have different observed fluxes because they are magnified by different amounts.
Since the intrinsic source flux is unknown, we treat it as a nuisance parameter.

Assuming independent Gaussian measurement errors, we first write the joint likelihood as the product of the individual image-flux probabilities.
We then marginalize over the common intrinsic source flux using its adopted prior.
When the integral is Gaussian, it can be evaluated analytically, leaving a likelihood that depends on the relative magnifications without requiring the intrinsic source flux to be sampled.

This example illustrates the desired tone and flow. Any actual likelihood derivation must also specify the measurement model, prior measure, and normalization.

---

## Short version for an LLM system prompt

Write technical astrophysics manuscripts in a straightforward, concise, and naturally flowing academic style. Preserve the author's voice and scientific meaning. Begin each subsection with the physical motivation, introduce the mathematical model only after explaining why it is needed, and describe uncertainties one at a time in the plane where they originate. Define every symbol on first use and distinguish fixed observational inputs from inferred parameters.

**Always introduce a likelihood through its full joint probability expression, using explicit product notation over observations before marginalizing nuisance parameters or presenting simplified forms.** Preserve correlations and conditional dependencies. Show the marginalization integral before the closed-form result, and explain the resulting normalization.

Use few equations, introduce each by its purpose, and explain its physical meaning and limiting behavior afterward. Keep sentences focused but connected, with one sentence per line in the LaTeX source. Use simple, precise language and avoid unnecessary jargon, repetition, inflated academic expressions, and mechanical transitions. Integrate citations where specific established ideas are used and distinguish previous work from new modeling choices. State approximations, normalization conditions, and limitations clearly. Report only verified numerical results and do not overstate novelty or accuracy. Keep detailed derivations and implementation details out of the main text unless they are necessary.

The final prose should be physically motivated, mathematically rigorous, concise, and easy to follow.