package state_abstraction_function;

import com.crawljax.browser.EmbeddedBrowser;
import com.crawljax.core.state.StateVertex;
import com.crawljax.core.state.StateVertexFactory;

public class Word2VecEmbeddingStateVertexFactory extends StateVertexFactory {
    private String classifierURI = "http://localhost:4242/classify";

//    creates object which will then make the comparison
    @Override
    public StateVertex newStateVertex(int id, String url, String name, String dom, String strippedDom, EmbeddedBrowser browser) {
        return new Word2VecEmbeddingStateVertexImpl(id, url, name, dom, strippedDom, classifierURI);
    }
}
