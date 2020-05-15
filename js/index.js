Kakao.init('f3b1e7ad3119c871d4c13360f05459e2');
Kakao.Link.createDefaultButton({
  container: '#kakao-link-btn',
  objectType: 'feed',
  content: {
    title: document.title,
    description: '내용, 주로 해시태그',
    imageUrl: document.images[1].src,
    link: {
      webUrl: document.location.href,
      mobileWebUrl: document.location.href,
    },
  },
  social: {
    likeCount: 286,
    commentCount: 45,
    sharedCount: 845,
  },
  buttons: [
    {
      title: 'Open!',
      link: {
        mobileWebUrl: document.location.href,
        webUrl: document.location.href,
      },
    },
  ],
});
