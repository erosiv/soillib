# soillib/todo

Live-Viewer:
- Tesselated Terrain Rendering
- SSAO in Viewer (Optional)

Relief-Shader:
- Iso-Line Image Output
- ???

TinyEngine needs improvements as well...

IO:
- TIFF Should be a variant, auto-detect size
- GEOTIFF Merging Program Should Exist. Should have a UI
	- Ideally Written in Python
- Images should be able to load at a sub-resolution
	- This means we don't have to load the whole thing!!!
	- OpenGL Textures should operate more nicely with these!!!

Tools:
- Write a Proper Merge Tool
- Write a Proper Relief-Shade Tool

Find out what the best techinque is for dynamic LOD in terrain.
Is there a benchmark somewhere?

Non-Tesselation Terrain Rendering, By Utilizing Chunking and Calling the render methods on sub-sections but with different index buffers depending on the level of detail that I want to render it at!

That's a good idea.
Then the only thing that I have to do is cull dynamically and compute the LOD dynamically.

Different LOD Indexing Schemes... Very Cool.

Basically that's exactly what I have to do. Eieiei.

Then whaT? SSAO?
Write about it? Vertex-pooled terrain rendering?

Soillib needs to be really cleaned up after this whole shabang.
But it's gonna end up having a ton of really nice features...
wow that's cool. Gotta make sure I move stuff into the right image stages.
