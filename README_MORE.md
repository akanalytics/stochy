

# <a name="relative_difference">Relative Differences</a>

Rather than specifying an objective function to be minimised (cost function), a relative difference function can be specified.
This feature is only available using the [stepwise](https://crates.io/crates/stepwise) API.

```text
df(x1, x2) ~ f(x2) - f(x1)
```
which permits use in cases where an abolute value of objective function is unavailable. Typically a game playing program would seek to minimise `-df` (and hence maximize `df`) where `x₁` and `x₂` represent game playing parameters, and the difference function df represents the outcome of a single game or a series of games.

```text
           / +1   x₂ win vs x₁ loss
df(x₁, x₂) =  0   drawn game
           \ -1   x₂ loss vs x₁ win
```








