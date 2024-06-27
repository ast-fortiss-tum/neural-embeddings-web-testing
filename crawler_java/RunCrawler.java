import com.crawljax.browser.EmbeddedBrowser;
import com.crawljax.core.CrawljaxRunner;
import com.crawljax.core.configuration.BrowserConfiguration;
import com.crawljax.core.configuration.CrawlRules;
import com.crawljax.core.configuration.CrawljaxConfiguration;
import com.crawljax.core.configuration.InputSpecification;
import com.crawljax.core.state.Identification;
import com.crawljax.forms.FormInput;
import com.crawljax.plugins.crawloverview.CrawlOverview;
import org.bytedeco.opencv.presets.opencv_core;
import state_abstraction_function.Word2VecEmbeddingStateVertexFactory;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;

import java.util.concurrent.TimeUnit;

public class RunCrawler {
    private static final long WAIT_TIME_AFTER_EVENT = 500;
    private static final long WAIT_TIME_AFTER_RELOAD = 500;

    private static final String URL = "http://localhost:3000/addressbook/";
    private static final String APP_NAME = "addressbook";
//    private static final String URL = "https://www.york.ac.uk/teaching/cws/wws/webpage1.html";

    public static void main(String[] args) throws Exception {

        WebDriverManager.chromedriver().clearDriverCache().clearResolutionCache();

        CrawljaxConfiguration.CrawljaxConfigurationBuilder builder = CrawljaxConfiguration.builderFor(URL);
//      1. set crawl rules
        builder.crawlRules().setFormFillMode(CrawlRules.FormFillMode.RANDOM);
        builder.crawlRules().clickDefaultElements();
        builder.crawlRules().crawlHiddenAnchors(true);
        builder.crawlRules().crawlFrames(false);
        builder.crawlRules().clickElementsInRandomOrder(false);
//      2. set max number of states
//        builder.setMaximumStates(maxStates);
        builder.setUnlimitedStates();
//      3. set max run time
        builder.setMaximumRunTime(3, TimeUnit.MINUTES);
//        builder.setUnlimitedRuntime();
//      4. set crawl depth
        builder.setUnlimitedCrawlDepth();
//      5. setup abstract function to be used
        builder.setStateVertexFactory(new Word2VecEmbeddingStateVertexFactory());
//      6. set timeouts
        builder.crawlRules().waitAfterReloadUrl(WAIT_TIME_AFTER_RELOAD, TimeUnit.MILLISECONDS);
        builder.crawlRules().waitAfterEvent(WAIT_TIME_AFTER_EVENT, TimeUnit.MILLISECONDS);
//      7. choose browser
        builder.setBrowserConfig(new BrowserConfiguration(EmbeddedBrowser.BrowserType.CHROME, 1));
//      8. add crawl overview
        builder.addPlugin(new CrawlOverview());

//      8.5 add input respective for the app
        builder.crawlRules().setInputSpec(Helper.setInputField(APP_NAME));

//      9. build crawler
        CrawljaxRunner crawljax = new CrawljaxRunner(builder.build());
//      10. run crawler
        crawljax.call();


    }

    private static class Helper {
        public static InputSpecification setInputField(String appName){
            InputSpecification input = new InputSpecification();
            switch (appName){
                case "addressbook":
                    // necessary for logging in
                    input.inputField(new FormInput(FormInput.InputType.INPUT, new Identification(Identification.How.name, "user"))).inputValues("admin");
                    input.inputField(new FormInput(FormInput.InputType.PASSWORD, new Identification(Identification.How.name, "pass"))).inputValues("secret");
                    System.out.println("Set input fields for addressbook");
                    break;
                case "claroline":
                    System.out.println("IMPLEMENT HANDLING FOR CLAROLINE");
                    break;
                default:
                    System.out.println(APP_NAME + " not implemented yet");
                    break;
            }
            return input;
        }
    }
}
