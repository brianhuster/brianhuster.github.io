// Import necessary libraries
const lunr = require('lunr');
require('lunr-languages/lunr.stemmer.support.js')(lunr);
require('lunr-languages/lunr.multi.js')(lunr);
require('../../assets/js/lunr.vi.js')(lunr);
const fs = require('fs');

// Load your documents from a JSON file
// This file should be generated by your Jekyll build process
const documents = require('../../_site/search.json');

// Build the index
const idx = lunr(function () {
    // this.use(lunr.multiLanguage('en'));
    this.use(lunr.vi);
    this.ref('id');
    this.field('title');
    this.field('body');

    documents.forEach(function (doc) {
        this.add(doc);
    }, this);
});

// Save the index to a file
fs.writeFileSync('./assets/server/search_index.json', JSON.stringify(idx));
console.log('Node : Search index generated successfully!');