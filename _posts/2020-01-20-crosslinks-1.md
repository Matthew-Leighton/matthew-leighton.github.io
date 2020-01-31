---
layout: post
title: "Cross-Links: Part 1"
date: 2020-01-20
---

<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>


This post is part 1 of a series where I'll be exploring the physics and mathematics of molecular cross-links in polymer networks.

### Sections to write:
* What are cross-links?
* Examples?
* What features do we want to model? 
 1. Probability Distributions
 2. Entropy
 3. Free Energies

Let's first consider the simplest example: a one-dimensional cross-link.

